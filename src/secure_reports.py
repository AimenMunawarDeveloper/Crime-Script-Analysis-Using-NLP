"""
secure_reports.py

Implements a manual "PKI + signed & encrypted reports" demo layer.

Each report is packaged as JSON (one per line):
{
  "report_id": "...",
  "ciphertext": "...",   # base64(XOR(report_bytes, K))
  "enc_key": "...",      # base64(RSA_encrypt_int(int_from_bytes(K), server_pub))
  "signature": "...",    # base64(RSA_sign_hash(hash(report), sender_priv))
  "public_key": { "n": "...", "e": "..." }   # base64-encoded ints (simulated certificate)
}

Receiver (analysis server):
- decrypt enc_key with server private key -> K
- XOR-decrypt ciphertext -> plaintext report
- recompute hash and verify signature using sender public key
"""

from __future__ import annotations

import json
import os
import secrets
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from manual_crypto import (
    RSAPrivateKey,
    RSAPublicKey,
    b64_to_int,
    canonicalize_text,
    hash_vishing_payload,
    djb2_64,
    int_to_b64,
    int_to_bytes,
    random_key_bytes,
    rsa_decrypt_int,
    rsa_encrypt_int,
    rsa_keygen,
    rsa_sign_hash,
    rsa_verify_hash,
    xor_stream,
    b64d,
    b64e,
)


def serialize_public_key(pub: RSAPublicKey) -> Dict[str, str]:
    return {"n": int_to_b64(pub.n), "e": int_to_b64(pub.e)}


def deserialize_public_key(d: Dict[str, str]) -> RSAPublicKey:
    return RSAPublicKey(n=b64_to_int(d["n"]), e=b64_to_int(d["e"]))


def serialize_private_key(priv: RSAPrivateKey) -> Dict[str, str]:
    # Store CRT params when present (more complex RSA private key).
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
    # v2-only: require CRT parameters (p, q, dp, dq, qinv) for the "advanced RSA" path.
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
    symmetric_key_len: int = 16,
    crime_type: str = "vishing",
) -> Dict[str, Any]:
    plaintext = canonicalize_text(report_text)
    # Sign a payload that's relevant to crime-script analysis:
    # bind the report to its ID + crime_type, not just raw text.
    h = hash_vishing_payload(report_id=report_id, report_text=plaintext, crime_type=crime_type)
    sig = rsa_sign_hash(h, sender_priv)

    k = random_key_bytes(symmetric_key_len)
    ct = xor_stream(plaintext.encode("utf-8", errors="replace"), k)

    # --- Use-case-specific RSA "key capsule" (vishing context binding) ---
    # Instead of RSA-encrypting raw K, we encrypt a structured capsule:
    #   [8-byte tag][1-byte key_len][key_bytes][random padding...]
    #
    # tag = djb2_64(f"{crime_type}|{report_id}") as 8 bytes
    # This binds the RSA-encrypted key to this *specific* report identity/context.
    # It prevents swapping enc_key between packages (a realistic dataset tampering attack).
    tag_int = djb2_64(f"{crime_type}|{str(report_id)}".encode("utf-8", errors="replace"))
    tag = int_to_bytes(tag_int, 8)
    klen = len(k)
    if not (1 <= klen <= 255):
        raise ValueError("symmetric key length out of range")
    header = tag + bytes([klen]) + k

    modulus_len = max(1, (server_pub.n.bit_length() + 7) // 8)
    # We'll prepend a 0x00 byte to guarantee capsule_int < n.
    # That means usable space is (modulus_len - 1) bytes.
    if len(header) >= (modulus_len - 1):
        raise ValueError("server RSA modulus too small for key capsule")

    # Add random padding to make the capsule look less repetitive (still manual, not OAEP).
    pad_len = (modulus_len - 1) - len(header)
    padding = secrets.token_bytes(pad_len) if pad_len > 0 else b""
    capsule = b"\x00" + header + padding
    capsule_int = int.from_bytes(capsule, byteorder="big")

    enc_k = rsa_encrypt_int(capsule_int, server_pub)

    return {
        "version": 2,
        "report_id": str(report_id),
        "crime_type": crime_type,
        "ciphertext": b64e(ct),
        "enc_key": int_to_b64(enc_k),
        "signature": int_to_b64(sig),
        # Keep key_len for transparency/debug (and backward compatibility with earlier v2 files).
        "key_len": int(len(k)),
        "public_key": serialize_public_key(sender_pub),
    }


def open_package(
    pkg: Dict[str, Any],
    server_priv: RSAPrivateKey,
) -> Tuple[str, str, bool]:
    """
    Returns (report_id, plaintext, verified).
    If decrypt fails, returns verified=False and plaintext="".
    """
    report_id = str(pkg.get("report_id", ""))
    try:
        if int(pkg.get("version", 0)) != 2:
            return report_id, "", False
        sender_pub = deserialize_public_key(pkg["public_key"])
        crime_type = str(pkg.get("crime_type", "vishing"))
        ct = b64d(pkg["ciphertext"])
        enc_k = b64_to_int(pkg["enc_key"])
        sig = b64_to_int(pkg["signature"])

        # RSA-decrypt the key capsule and extract the XOR key.
        capsule_int = rsa_decrypt_int(enc_k, server_priv)
        modulus_len = max(1, (server_priv.n.bit_length() + 7) // 8)
        capsule = int_to_bytes(capsule_int, modulus_len)

        tag_expected_int = djb2_64(f"{crime_type}|{str(report_id)}".encode("utf-8", errors="replace"))
        tag_expected = int_to_bytes(tag_expected_int, 8)
        # capsule layout: [0x00][8-byte tag][1-byte key_len][key...][padding...]
        if capsule[:1] != b"\x00":
            return report_id, "", False
        tag_got = capsule[1:9]
        if tag_got != tag_expected:
            return report_id, "", False
        k_len = capsule[9]
        if k_len <= 0:
            return report_id, "", False
        k = capsule[10 : 10 + k_len]
        if len(k) != k_len:
            return report_id, "", False

        pt_bytes = xor_stream(ct, k)
        plaintext = pt_bytes.decode("utf-8", errors="replace")
        plaintext = canonicalize_text(plaintext)

        h = hash_vishing_payload(report_id=report_id, report_text=plaintext, crime_type=crime_type)
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


def load_public_key_json(path: str) -> RSAPublicKey:
    return deserialize_public_key(json.loads(read_text_file(path)))


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
    """
    Returns (verified_rows, verified_count, total_count).
    Each verified row has: {"report_id": str, "plaintext": str}
    """
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
    # Simple demo CLI:
    # python src/secure_reports.py
    #   - Generates server + sender keys
    #   - Packages the raw dataset into Data Set/secure_reports.jsonl
    #   - Writes keys into Data Set/
    #
    # This keeps the main NLP pipeline unchanged unless secure reports are present.
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Generate demo secure report packages (JSONL)")
    parser.add_argument("--csv", default=None, help="Path to scam_raw_dataset.csv")
    parser.add_argument("--out", default=None, help="Path to write secure_reports.jsonl")
    parser.add_argument("--text-col", default="incident_description", help="Column to package as report text")
    parser.add_argument("--id-col", default="submission_id", help="Column to use as report_id (fallback: row index)")
    parser.add_argument("--limit", type=int, default=200, help="Max rows to package (for speed)")
    parser.add_argument("--server-bits", type=int, default=256, help="RSA bits for server key")
    parser.add_argument("--sender-bits", type=int, default=256, help="RSA bits for sender key")
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

