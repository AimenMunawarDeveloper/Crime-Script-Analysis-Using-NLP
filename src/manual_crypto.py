from __future__ import annotations

import base64
import secrets
from dataclasses import dataclass
from typing import Optional, Tuple


def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def modinv(a: int, m: int) -> int:
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m


def modexp(base: int, exp: int, mod: int) -> int:
    if mod <= 0:
        raise ValueError("mod must be positive")
    result = 1
    base %= mod
    e = exp
    while e > 0:
        if e & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        e >>= 1
    return result


def _is_probable_prime(n: int, rounds: int = 10) -> bool:
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small_primes:
        return True
    if any((n % p) == 0 for p in small_primes):
        return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = modexp(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _r in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True


def generate_prime(bits: int) -> int:
    if bits < 16:
        raise ValueError("bits too small for demo prime generation")
    while True:
        candidate = secrets.randbits(bits)
        candidate |= (1 << (bits - 1)) | 1
        if _is_probable_prime(candidate):
            return candidate


def canonicalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.strip() for line in s.split("\n"))
    return s.strip()


def djb2_64(data: bytes) -> int:
    h = 5381
    for b in data:
        h = ((h << 5) + h) ^ b
        h &= 0xFFFFFFFFFFFFFFFF
    return h


def hash_vishing_payload(report_id: str, report_text: str, crime_type: str = "vishing") -> int:
    payload = f"crime_type={crime_type}\nreport_id={str(report_id).strip()}\ntext={canonicalize_text(report_text)}"
    return djb2_64(payload.encode("utf-8", errors="replace"))


@dataclass(frozen=True)
class RSAPublicKey:
    n: int
    e: int


@dataclass(frozen=True)
class RSAPrivateKey:
    n: int
    d: int
    p: Optional[int] = None
    q: Optional[int] = None
    dp: Optional[int] = None
    dq: Optional[int] = None
    qinv: Optional[int] = None


def rsa_keygen(bits: int = 256, e: int = 65537) -> Tuple[RSAPublicKey, RSAPrivateKey]:
    if bits < 128:
        raise ValueError("bits too small; use >= 128 for demo")
    half = bits // 2
    while True:
        p = generate_prime(half)
        q = generate_prime(half)
        if p == q:
            continue
        n = p * q
        phi = (p - 1) * (q - 1)
        if phi % e == 0:
            continue
        try:
            d = modinv(e, phi)
        except ValueError:
            continue
        dp = d % (p - 1)
        dq = d % (q - 1)
        qinv = modinv(q, p)
        return RSAPublicKey(n=n, e=e), RSAPrivateKey(n=n, d=d, p=p, q=q, dp=dp, dq=dq, qinv=qinv)


def rsa_encrypt_int(m: int, pub: RSAPublicKey) -> int:
    if not (0 <= m < pub.n):
        raise ValueError("message integer out of range for RSA modulus")
    return modexp(m, pub.e, pub.n)


def rsa_decrypt_int(c: int, priv: RSAPrivateKey) -> int:
    if not (0 <= c < priv.n):
        raise ValueError("ciphertext integer out of range for RSA modulus")
    if priv.p and priv.q and priv.dp is not None and priv.dq is not None and priv.qinv is not None:
        p = priv.p
        q = priv.q
        m1 = modexp(c % p, priv.dp, p)
        m2 = modexp(c % q, priv.dq, q)
        h = (priv.qinv * (m1 - m2)) % p
        return m2 + h * q
    return modexp(c, priv.d, priv.n)


def rsa_sign_hash(hash_int: int, priv: RSAPrivateKey) -> int:
    h = hash_int % priv.n
    return rsa_decrypt_int(h, priv)


def rsa_verify_hash(hash_int: int, signature_int: int, pub: RSAPublicKey) -> bool:
    h = hash_int % pub.n
    v = modexp(signature_int, pub.e, pub.n)
    return v == h


def xor_stream(data: bytes, key: bytes) -> bytes:
    if not key:
        raise ValueError("key must not be empty")
    out = bytearray(len(data))
    for i, b in enumerate(data):
        out[i] = b ^ key[i % len(key)]
    return bytes(out)


def random_key_bytes(length: int = 16) -> bytes:
    if length <= 0:
        raise ValueError("length must be positive")
    return secrets.token_bytes(length)


def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def int_to_b64(i: int) -> str:
    if i < 0:
        raise ValueError("int must be non-negative")
    nbytes = max(1, (i.bit_length() + 7) // 8)
    return b64e(i.to_bytes(nbytes, byteorder="big"))


def b64_to_int(s: str) -> int:
    b = b64d(s)
    return int.from_bytes(b, byteorder="big")


def int_to_bytes(i: int, length: Optional[int] = None) -> bytes:
    if i < 0:
        raise ValueError("int must be non-negative")
    min_len = max(1, (i.bit_length() + 7) // 8)
    if length is None:
        length = min_len
    if length < min_len:
        raise ValueError("length too small to hold integer")
    return i.to_bytes(length, byteorder="big")

