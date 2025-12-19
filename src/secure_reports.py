from __future__ import annotations

import json
import os
import secrets
import time
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from manual_crypto import (
    RSAPrivateKey,
    RSAPublicKey,
    aes256_cbc_decrypt,
    aes256_cbc_encrypt,
    b64_to_int,
    canonicalize_text,
    generate_vishing_iv,
    hash_vishing_payload,
    hmac_sha256,
    int_to_b64,
    int_to_bytes,
    oaep_decode,
    oaep_encode,
    random_key_bytes,
    rsa_decrypt_int,
    rsa_encrypt_int,
    rsa_keygen,
    rsa_sign_hash,
    rsa_verify_hash,
    sha256_hash,
    b64d,
    b64e,
)


def serialize_public_key(pub: RSAPublicKey) -> Dict[str, str]:
    return {"n": int_to_b64(pub.n), "e": int_to_b64(pub.e)}


def deserialize_public_key(d: Dict[str, str]) -> RSAPublicKey:
    return RSAPublicKey(n=b64_to_int(d["n"]), e=b64_to_int(d["e"]))


def serialize_private_key(priv: RSAPrivateKey) -> Dict[str, str]:
    out: Dict[str, str] = {"n": int_to_b64(priv.n), "d": int_to_b64(priv.d)}
    if priv.p is not None:
        out["p"] = int_to_b64(priv.p)
    if priv.q is not None:
        out["q"] = int_to_b64(priv.q)
    if priv.dp is not None:
        out["dp"] = int_to_b64(priv.dp)
    if priv.dq is not None:
        out["dq"] = int_to_b64(priv.dq)
    if priv.qinv is not None:
        out["qinv"] = int_to_b64(priv.qinv)
    return out


def deserialize_private_key(d: Dict[str, str]) -> RSAPrivateKey:
    required = ("n", "d", "p", "q", "dp", "dq", "qinv")
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"Missing required RSA private key fields: {missing}")
    return RSAPrivateKey(
        n=b64_to_int(d["n"]),
        d=b64_to_int(d["d"]),
        p=b64_to_int(d["p"]),
        q=b64_to_int(d["q"]),
        dp=b64_to_int(d["dp"]),
        dq=b64_to_int(d["dq"]),
        qinv=b64_to_int(d["qinv"]),
    )


def create_package(
    report_id: str,
    report_text: str,
    sender_priv: RSAPrivateKey,
    sender_pub: RSAPublicKey,
    server_pub: RSAPublicKey,
    symmetric_key_len: int = 32,
    crime_type: str = "vishing",
) -> Dict[str, Any]:
    plaintext = canonicalize_text(report_text)
    timestamp = int(time.time())
    
    h = hash_vishing_payload(
        report_id=report_id, 
        report_text=plaintext, 
        crime_type=crime_type,
        timestamp=timestamp
    )
    sig = rsa_sign_hash(h, sender_priv)

    k = random_key_bytes(symmetric_key_len)
    if len(k) != 32:
        raise ValueError("AES-256 requires 32-byte key")
    
    iv = generate_vishing_iv(report_id=report_id, crime_type=crime_type, timestamp=timestamp)
    
    pt_bytes = plaintext.encode("utf-8", errors="replace")
    ct = aes256_cbc_encrypt(pt_bytes, k, iv)
    
    hmac_key_material = k + f"{crime_type}|{report_id}|{timestamp}".encode('utf-8')
    hmac_key = sha256_hash(hmac_key_material)[:32]
    hmac_value = hmac_sha256(hmac_key, ct)

    oaep_label = f"vishing|{crime_type}|{report_id}".encode('utf-8')
    
    key_material = iv + k
    
    modulus_len = max(1, (server_pub.n.bit_length() + 7) // 8)
    oaep_overhead = 2 * 32 + 2
    max_message_len = modulus_len - oaep_overhead
    if len(key_material) > max_message_len:
        raise ValueError(
            f"Key material ({len(key_material)} bytes) too large for RSA modulus "
            f"({modulus_len} bytes) with OAEP (needs {oaep_overhead} bytes overhead). "
            f"Maximum message length: {max_message_len} bytes. "
            f"Consider using larger RSA key size (e.g., --server-bits 1024 or 2048)."
        )
    
    encoded_key = oaep_encode(key_material, oaep_label, server_pub.n.bit_length())
    encoded_key_int = int.from_bytes(encoded_key, byteorder="big")
    
    if encoded_key_int >= server_pub.n:
        raise ValueError("OAEP encoded key too large for RSA modulus")
    
    enc_k = rsa_encrypt_int(encoded_key_int, server_pub)

    return {
        "version": 3,
        "report_id": str(report_id),
        "crime_type": crime_type,
        "timestamp": timestamp,
        "ciphertext": b64e(ct),
        "enc_key": int_to_b64(enc_k),
        "signature": int_to_b64(sig),
        "hmac": b64e(hmac_value),
        "public_key": serialize_public_key(sender_pub),
    }


def open_package(
    pkg: Dict[str, Any],
    server_priv: RSAPrivateKey,
) -> Tuple[str, str, bool]:
    report_id = str(pkg.get("report_id", ""))
    try:
        version = int(pkg.get("version", 0))
        if version != 3:
            return report_id, "", False
        
        sender_pub = deserialize_public_key(pkg["public_key"])
        crime_type = str(pkg.get("crime_type", "vishing"))
        ct = b64d(pkg["ciphertext"])
        enc_k = b64_to_int(pkg["enc_key"])
        sig = b64_to_int(pkg["signature"])
        timestamp = pkg.get("timestamp")
        
        if timestamp is None:
            return report_id, "", False
        
        capsule_int = rsa_decrypt_int(enc_k, server_priv)
        modulus_len = max(1, (server_priv.n.bit_length() + 7) // 8)
        capsule = int_to_bytes(capsule_int, modulus_len)
        
        oaep_label = f"vishing|{crime_type}|{report_id}".encode('utf-8')
        try:
            key_material = oaep_decode(capsule, oaep_label, server_priv.n.bit_length())
        except ValueError:
            return report_id, "", False
        
        if len(key_material) < 48:
            return report_id, "", False
        
        iv = key_material[0:16]
        k = key_material[16:48]
        
        if len(k) != 32:
            return report_id, "", False
        
        hmac_key_material = k + f"{crime_type}|{report_id}|{timestamp}".encode('utf-8')
        hmac_key = sha256_hash(hmac_key_material)[:32]
        
        hmac_expected = b64d(pkg.get("hmac", ""))
        if not hmac_expected:
            return report_id, "", False
        
        hmac_calculated = hmac_sha256(hmac_key, ct)
        if hmac_calculated != hmac_expected:
            return report_id, "", False
        
        try:
            pt_bytes = aes256_cbc_decrypt(ct, k, iv)
        except ValueError:
            return report_id, "", False
        
        plaintext = pt_bytes.decode("utf-8", errors="replace")
        plaintext = canonicalize_text(plaintext)
        
        h = hash_vishing_payload(
            report_id=report_id, 
            report_text=plaintext, 
            crime_type=crime_type,
            timestamp=timestamp
        )
        verified = rsa_verify_hash(h, sig, sender_pub)
        
        return report_id, plaintext, verified
    except Exception:
        return report_id, "", False


def dumps_jsonl(packages: Iterable[Dict[str, Any]]) -> str:
    return "\n".join(json.dumps(p, ensure_ascii=False) for p in packages) + "\n"


def loads_jsonl(text: str) -> Iterator[Dict[str, Any]]:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def generate_demo_identities(
    server_bits: int = 256,
    sender_bits: int = 256,
) -> Tuple[Tuple[RSAPublicKey, RSAPrivateKey], Tuple[RSAPublicKey, RSAPrivateKey]]:
    server_pub, server_priv = rsa_keygen(bits=server_bits)
    sender_pub, sender_priv = rsa_keygen(bits=sender_bits)
    return (server_pub, server_priv), (sender_pub, sender_priv)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text_file(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_keypair_json(pub: RSAPublicKey, priv: RSAPrivateKey, pub_path: str, priv_path: str) -> None:
    write_text_file(pub_path, json.dumps(serialize_public_key(pub), indent=2))
    write_text_file(priv_path, json.dumps(serialize_private_key(priv), indent=2))


def load_private_key_json(path: str) -> RSAPrivateKey:
    return deserialize_private_key(json.loads(read_text_file(path)))


def load_secure_reports_jsonl(path: str) -> List[Dict[str, Any]]:
    return list(loads_jsonl(read_text_file(path)))


def save_secure_reports_jsonl(path: str, packages: Iterable[Dict[str, Any]]) -> None:
    write_text_file(path, dumps_jsonl(packages))


def decrypt_and_verify_packages(
    packages: Iterable[Dict[str, Any]],
    server_priv: RSAPrivateKey,
) -> Tuple[List[Dict[str, Any]], int, int]:
    verified_rows: List[Dict[str, Any]] = []
    total = 0
    verified = 0
    for pkg in packages:
        total += 1
        report_id, plaintext, ok = open_package(pkg, server_priv)
        if ok:
            verified += 1
            verified_rows.append({"report_id": report_id, "plaintext": plaintext})
    return verified_rows, verified, total


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Generate demo secure report packages (JSONL)")
    parser.add_argument("--csv", default=None, help="Path to scam_raw_dataset.csv")
    parser.add_argument("--out", default=None, help="Path to write secure_reports.jsonl")
    parser.add_argument("--text-col", default="incident_description", help="Column to package as report text")
    parser.add_argument("--id-col", default="submission_id", help="Column to use as report_id (fallback: row index)")
    parser.add_argument("--limit", type=int, default=200, help="Max rows to package (for speed)")
    parser.add_argument("--server-bits", type=int, default=1024, help="RSA bits for server key (minimum 1024 for OAEP with 48-byte key material)")
    parser.add_argument("--sender-bits", type=int, default=1024, help="RSA bits for sender key")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "Data Set")

    csv_path = args.csv or os.path.join(data_dir, "scam_raw_dataset.csv")
    out_path = args.out or os.path.join(data_dir, "secure_reports.jsonl")

    server_pub_path = os.path.join(data_dir, "server_rsa_public.json")
    server_priv_path = os.path.join(data_dir, "server_rsa_private.json")
    sender_pub_path = os.path.join(data_dir, "sender_rsa_public.json")
    sender_priv_path = os.path.join(data_dir, "sender_rsa_private.json")

    (server_pub, server_priv), (sender_pub, sender_priv) = generate_demo_identities(
        server_bits=args.server_bits,
        sender_bits=args.sender_bits,
    )

    save_keypair_json(server_pub, server_priv, server_pub_path, server_priv_path)
    save_keypair_json(sender_pub, sender_priv, sender_pub_path, sender_priv_path)

    df = pd.read_csv(csv_path)
    if args.text_col not in df.columns:
        raise SystemExit(f"Missing text column: {args.text_col}")

    packages: List[Dict[str, Any]] = []
    limit = min(args.limit, len(df))
    for i in range(limit):
        row = df.iloc[i]
        report_text = "" if pd.isna(row[args.text_col]) else str(row[args.text_col])
        if args.id_col in df.columns and not pd.isna(row[args.id_col]):
            report_id = str(row[args.id_col])
        else:
            report_id = str(i)
        packages.append(
            create_package(
                report_id=report_id,
                report_text=report_text,
                sender_priv=sender_priv,
                sender_pub=sender_pub,
                server_pub=server_pub,
            )
        )

    save_secure_reports_jsonl(out_path, packages)
    print(f"Wrote {len(packages)} secure packages to: {out_path}")
    print(f"Wrote server keys to: {server_pub_path} and {server_priv_path}")
    print(f"Wrote sender keys to: {sender_pub_path} and {sender_priv_path}")

