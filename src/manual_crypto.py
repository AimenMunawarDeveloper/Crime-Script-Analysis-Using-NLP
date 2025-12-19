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


def _right_rotate(value: int, amount: int) -> int:
    return ((value >> amount) | (value << (32 - amount))) & 0xFFFFFFFF


def sha256_hash(data: bytes) -> bytes:
    h = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    ]
    
    k = [
        0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
        0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
        0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
        0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
        0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
        0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
        0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x6C44198C,
        0x682E6FF3, 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7,
        0xC67178F2
    ]
    
    original_length = len(data) * 8
    data = bytearray(data)
    data.append(0x80)
    
    while (len(data) * 8) % 512 != 448:
        data.append(0x00)
    
    data.extend(original_length.to_bytes(8, byteorder='big'))
    
    for chunk_start in range(0, len(data), 64):
        chunk = data[chunk_start:chunk_start + 64]
        
        w = [0] * 64
        for i in range(16):
            w[i] = int.from_bytes(chunk[i*4:(i+1)*4], byteorder='big')
        
        for i in range(16, 64):
            s0 = _right_rotate(w[i-15], 7) ^ _right_rotate(w[i-15], 18) ^ (w[i-15] >> 3)
            s1 = _right_rotate(w[i-2], 17) ^ _right_rotate(w[i-2], 19) ^ (w[i-2] >> 10)
            w[i] = (w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF
        
        a, b, c, d, e, f, g, h_temp = h
        
        for i in range(64):
            S1 = _right_rotate(e, 6) ^ _right_rotate(e, 11) ^ _right_rotate(e, 25)
            ch = (e & f) ^ ((~e) & g)
            temp1 = (h_temp + S1 + ch + k[i] + w[i]) & 0xFFFFFFFF
            S0 = _right_rotate(a, 2) ^ _right_rotate(a, 13) ^ _right_rotate(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & 0xFFFFFFFF
            
            h_temp = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF
        
        h[0] = (h[0] + a) & 0xFFFFFFFF
        h[1] = (h[1] + b) & 0xFFFFFFFF
        h[2] = (h[2] + c) & 0xFFFFFFFF
        h[3] = (h[3] + d) & 0xFFFFFFFF
        h[4] = (h[4] + e) & 0xFFFFFFFF
        h[5] = (h[5] + f) & 0xFFFFFFFF
        h[6] = (h[6] + g) & 0xFFFFFFFF
        h[7] = (h[7] + h_temp) & 0xFFFFFFFF
    
    digest = bytearray()
    for val in h:
        digest.extend(val.to_bytes(4, byteorder='big'))
    
    return bytes(digest)


def sha256_hash_to_int(data: bytes) -> int:
    return int.from_bytes(sha256_hash(data), byteorder='big')


def hash_vishing_payload(report_id: str, report_text: str, crime_type: str = "vishing", timestamp: Optional[int] = None) -> int:
    if timestamp is None:
        import time
        timestamp = int(time.time())
    
    payload = f"crime_type={crime_type}\nreport_id={str(report_id).strip()}\ntimestamp={timestamp}\ntext={canonicalize_text(report_text)}"
    return sha256_hash_to_int(payload.encode("utf-8", errors="replace"))




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


def rsa_encrypt_oaep(message: bytes, pub: RSAPublicKey, label: bytes = b"") -> bytes:
    encoded = oaep_encode(message, label, pub.n.bit_length())
    encoded_int = int.from_bytes(encoded, byteorder="big")
    if encoded_int >= pub.n:
        raise ValueError("Encoded message too large for RSA modulus")
    ciphertext_int = rsa_encrypt_int(encoded_int, pub)
    k = (pub.n.bit_length() + 7) // 8
    return int_to_bytes(ciphertext_int, k)


def rsa_decrypt_oaep(ciphertext: bytes, priv: RSAPrivateKey, label: bytes = b"") -> bytes:
    k = (priv.n.bit_length() + 7) // 8
    if len(ciphertext) != k:
        raise ValueError("Ciphertext length mismatch")
    ciphertext_int = int.from_bytes(ciphertext, byteorder="big")
    plaintext_int = rsa_decrypt_int(ciphertext_int, priv)
    plaintext_encoded = int_to_bytes(plaintext_int, k)
    return oaep_decode(plaintext_encoded, label, priv.n.bit_length())


def rsa_sign_hash(hash_int: int, priv: RSAPrivateKey) -> int:
    h = hash_int % priv.n
    return rsa_decrypt_int(h, priv)


def rsa_verify_hash(hash_int: int, signature_int: int, pub: RSAPublicKey) -> bool:
    h = hash_int % pub.n
    v = modexp(signature_int, pub.e, pub.n)
    return v == h


_AES_SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

_AES_INV_SBOX = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
]

_AES_RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a]


def _aes_sub_word(word: int) -> int:
    return (_AES_SBOX[(word >> 24) & 0xFF] << 24) | \
           (_AES_SBOX[(word >> 16) & 0xFF] << 16) | \
           (_AES_SBOX[(word >> 8) & 0xFF] << 8) | \
           _AES_SBOX[word & 0xFF]


def _aes_rot_word(word: int) -> int:
    return ((word << 8) | (word >> 24)) & 0xFFFFFFFF


def _aes_key_expansion(key: bytes) -> List[int]:
    if len(key) != 32:
        raise ValueError("AES-256 requires 32-byte key")
    
    n_rounds = 14
    n_words = 8
    
    w = [0] * (4 * (n_rounds + 1))
    
    for i in range(n_words):
        w[i] = int.from_bytes(key[i*4:(i+1)*4], byteorder='big')
    
    for i in range(n_words, 4 * (n_rounds + 1)):
        temp = w[i - 1]
        if i % n_words == 0:
            temp = _aes_sub_word(_aes_rot_word(temp)) ^ (_AES_RCON[(i // n_words) - 1] << 24)
        elif i % n_words == 4:
            temp = _aes_sub_word(temp)
        w[i] = w[i - n_words] ^ temp
    
    return w


def _aes_add_round_key(state: List[List[int]], round_key: List[int]):
    for i in range(4):
        state[i] = [
            state[i][0] ^ ((round_key[i] >> 24) & 0xFF),
            state[i][1] ^ ((round_key[i] >> 16) & 0xFF),
            state[i][2] ^ ((round_key[i] >> 8) & 0xFF),
            state[i][3] ^ (round_key[i] & 0xFF)
        ]


def _aes_sub_bytes(state: List[List[int]], inv: bool = False):
    sbox = _AES_INV_SBOX if inv else _AES_SBOX
    for i in range(4):
        for j in range(4):
            state[i][j] = sbox[state[i][j]]


def _aes_shift_rows(state: List[List[int]], inv: bool = False):
    if inv:
        state[1] = [state[1][3], state[1][0], state[1][1], state[1][2]]
        state[2] = [state[2][2], state[2][3], state[2][0], state[2][1]]
        state[3] = [state[3][1], state[3][2], state[3][3], state[3][0]]
    else:
        state[1] = [state[1][1], state[1][2], state[1][3], state[1][0]]
        state[2] = [state[2][2], state[2][3], state[2][0], state[2][1]]
        state[3] = [state[3][3], state[3][0], state[3][1], state[3][2]]


def _aes_xtime(x: int) -> int:
    return ((x << 1) ^ (0x1B if (x & 0x80) else 0)) & 0xFF


def _aes_multiply(x: int, y: int) -> int:
    result = 0
    for i in range(8):
        if y & 1:
            result ^= x
        x = _aes_xtime(x)
        y >>= 1
    return result & 0xFF


def _aes_mix_columns(state: List[List[int]], inv: bool = False):
    for i in range(4):
        if inv:
            s0 = _aes_multiply(state[0][i], 0x0E) ^ _aes_multiply(state[1][i], 0x0B) ^ \
                 _aes_multiply(state[2][i], 0x0D) ^ _aes_multiply(state[3][i], 0x09)
            s1 = _aes_multiply(state[0][i], 0x09) ^ _aes_multiply(state[1][i], 0x0E) ^ \
                 _aes_multiply(state[2][i], 0x0B) ^ _aes_multiply(state[3][i], 0x0D)
            s2 = _aes_multiply(state[0][i], 0x0D) ^ _aes_multiply(state[1][i], 0x09) ^ \
                 _aes_multiply(state[2][i], 0x0E) ^ _aes_multiply(state[3][i], 0x0B)
            s3 = _aes_multiply(state[0][i], 0x0B) ^ _aes_multiply(state[1][i], 0x0D) ^ \
                 _aes_multiply(state[2][i], 0x09) ^ _aes_multiply(state[3][i], 0x0E)
        else:
            s0 = _aes_multiply(state[0][i], 2) ^ _aes_multiply(state[1][i], 3) ^ state[2][i] ^ state[3][i]
            s1 = state[0][i] ^ _aes_multiply(state[1][i], 2) ^ _aes_multiply(state[2][i], 3) ^ state[3][i]
            s2 = state[0][i] ^ state[1][i] ^ _aes_multiply(state[2][i], 2) ^ _aes_multiply(state[3][i], 3)
            s3 = _aes_multiply(state[0][i], 3) ^ state[1][i] ^ state[2][i] ^ _aes_multiply(state[3][i], 2)
        state[0][i] = s0 & 0xFF
        state[1][i] = s1 & 0xFF
        state[2][i] = s2 & 0xFF
        state[3][i] = s3 & 0xFF


def aes256_encrypt_block(block: bytes, key: bytes) -> bytes:
    if len(block) != 16:
        raise ValueError("Block must be 16 bytes")
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes for AES-256")
    
    round_keys = _aes_key_expansion(key)
    
    state = [[0] * 4 for _ in range(4)]
    for c in range(4):
        for r in range(4):
            state[r][c] = block[r + 4 * c]
    
    _aes_add_round_key(state, [round_keys[i] for i in range(4)])
    
    for round_num in range(1, 14):
        _aes_sub_bytes(state)
        _aes_shift_rows(state)
        _aes_mix_columns(state)
        _aes_add_round_key(state, [round_keys[4*round_num + i] for i in range(4)])
    
    _aes_sub_bytes(state)
    _aes_shift_rows(state)
    _aes_add_round_key(state, [round_keys[56 + i] for i in range(4)])
    
    output = bytearray(16)
    for c in range(4):
        for r in range(4):
            output[r + 4 * c] = state[r][c]
    
    return bytes(output)


def aes256_decrypt_block(block: bytes, key: bytes) -> bytes:
    if len(block) != 16:
        raise ValueError("Block must be 16 bytes")
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes for AES-256")
    
    round_keys = _aes_key_expansion(key)
    
    state = [[0] * 4 for _ in range(4)]
    for c in range(4):
        for r in range(4):
            state[r][c] = block[r + 4 * c]
    
    _aes_add_round_key(state, [round_keys[56 + i] for i in range(4)])
    
    for round_num in range(13, 0, -1):
        _aes_shift_rows(state, inv=True)
        _aes_sub_bytes(state, inv=True)
        _aes_add_round_key(state, [round_keys[4*round_num + i] for i in range(4)])
        _aes_mix_columns(state, inv=True)
    
    _aes_shift_rows(state, inv=True)
    _aes_sub_bytes(state, inv=True)
    _aes_add_round_key(state, [round_keys[i] for i in range(4)])
    
    output = bytearray(16)
    for c in range(4):
        for r in range(4):
            output[r + 4 * c] = state[r][c]
    
    return bytes(output)


def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    padding = bytes([pad_len] * pad_len)
    return data + padding


def pkcs7_unpad(data: bytes) -> bytes:
    if len(data) == 0:
        raise ValueError("Cannot unpad empty data")
    pad_len = data[-1]
    if pad_len < 1 or pad_len > 16 or pad_len > len(data):
        raise ValueError("Invalid padding")
    if not all(b == pad_len for b in data[-pad_len:]):
        raise ValueError("Invalid padding")
    return data[:-pad_len]


def aes256_cbc_encrypt(plaintext: bytes, key: bytes, iv: bytes) -> bytes:
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes for AES-256")
    if len(iv) != 16:
        raise ValueError("IV must be 16 bytes")
    
    padded = pkcs7_pad(plaintext, 16)
    
    ciphertext = bytearray()
    prev_block = iv
    
    for i in range(0, len(padded), 16):
        block = padded[i:i+16]
        xored = bytes(a ^ b for a, b in zip(block, prev_block))
        encrypted = aes256_encrypt_block(xored, key)
        ciphertext.extend(encrypted)
        prev_block = encrypted
    
    return bytes(ciphertext)


def aes256_cbc_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes for AES-256")
    if len(iv) != 16:
        raise ValueError("IV must be 16 bytes")
    if len(ciphertext) % 16 != 0:
        raise ValueError("Ciphertext length must be multiple of 16")
    
    plaintext = bytearray()
    prev_block = iv
    
    for i in range(0, len(ciphertext), 16):
        block = ciphertext[i:i+16]
        decrypted = aes256_decrypt_block(block, key)
        xored = bytes(a ^ b for a, b in zip(decrypted, prev_block))
        plaintext.extend(xored)
        prev_block = block
    
    return pkcs7_unpad(bytes(plaintext))


def hmac_sha256(key: bytes, message: bytes) -> bytes:
    block_size = 64
    
    if len(key) > block_size:
        key = sha256_hash(key)
    
    if len(key) < block_size:
        key = key + bytes(block_size - len(key))
    
    ipad = bytes(0x36 for _ in range(block_size))
    opad = bytes(0x5C for _ in range(block_size))
    
    inner_key = bytes(a ^ b for a, b in zip(key, ipad))
    inner_hash = sha256_hash(inner_key + message)
    
    outer_key = bytes(a ^ b for a, b in zip(key, opad))
    hmac_result = sha256_hash(outer_key + inner_hash)
    
    return hmac_result


def oaep_encode(message: bytes, label: bytes, n_bits: int) -> bytes:
    h_len = 32
    k = (n_bits + 7) // 8
    
    if len(message) > k - 2 * h_len - 2:
        raise ValueError("Message too long for OAEP encoding")
    
    seed = secrets.token_bytes(h_len)
    
    lhash = sha256_hash(label)
    ps_len = k - len(message) - 2 * h_len - 2
    db = lhash + bytes(ps_len) + b'\x01' + message
    
    db_mask = _mgf1(seed, k - h_len - 1, h_len)
    masked_db = bytes(a ^ b for a, b in zip(db, db_mask))
    
    seed_mask = _mgf1(masked_db, h_len, h_len)
    masked_seed = bytes(a ^ b for a, b in zip(seed, seed_mask))
    
    em = b'\x00' + masked_seed + masked_db
    
    return em


def oaep_decode(encoded: bytes, label: bytes, n_bits: int) -> bytes:
    h_len = 32
    k = (n_bits + 7) // 8
    
    if len(encoded) != k or k < 2 * h_len + 2:
        raise ValueError("Invalid OAEP encoding length")
    
    if encoded[0] != 0:
        raise ValueError("Invalid OAEP encoding: leading byte not zero")
    
    masked_seed = encoded[1:1+h_len]
    masked_db = encoded[1+h_len:]
    
    seed_mask = _mgf1(masked_db, h_len, h_len)
    seed = bytes(a ^ b for a, b in zip(masked_seed, seed_mask))
    
    db_mask = _mgf1(seed, k - h_len - 1, h_len)
    db = bytes(a ^ b for a, b in zip(masked_db, db_mask))
    
    lhash = sha256_hash(label)
    if db[:h_len] != lhash:
        raise ValueError("OAEP label mismatch")
    
    ps_end = h_len
    while ps_end < len(db) and db[ps_end] == 0:
        ps_end += 1
    
    if ps_end >= len(db) or db[ps_end] != 0x01:
        raise ValueError("Invalid OAEP padding")
    
    message = db[ps_end + 1:]
    return message


def _mgf1(seed: bytes, mask_len: int, h_len: int) -> bytes:
    t = bytearray()
    counter = 0
    
    while len(t) < mask_len:
        c = counter.to_bytes(4, byteorder='big')
        t.extend(sha256_hash(seed + c))
        counter += 1
    
    return bytes(t[:mask_len])


def generate_vishing_iv(report_id: str, crime_type: str = "vishing", timestamp: Optional[int] = None) -> bytes:
    if timestamp is None:
        import time
        timestamp = int(time.time())
    
    iv_material = f"vishing_iv|{crime_type}|{report_id}|{timestamp}".encode('utf-8')
    
    iv_hash = sha256_hash(iv_material)
    return iv_hash[:16]


def generate_vishing_nonce(report_id: str, crime_type: str = "vishing", length: int = 16) -> bytes:
    import time
    timestamp = int(time.time())
    
    base = f"vishing_nonce|{crime_type}|{report_id}|{timestamp}".encode('utf-8')
    base_hash = sha256_hash(base)
    
    random_part = secrets.token_bytes(length)
    
    combined = base_hash + random_part
    nonce = sha256_hash(combined)
    
    return nonce[:length]




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

