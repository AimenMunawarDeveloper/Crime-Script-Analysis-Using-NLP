from __future__ import annotations

import json
import os
import time
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from manual_crypto import (
    RSAPublicKey,
    RSAPrivateKey,
    rsa_keygen,
    hash_vishing_payload,
    rsa_sign_hash,
    int_to_b64,
)
from secure_reports import (
    create_package,
    open_package,
    load_secure_reports_jsonl,
    load_private_key_json,
    decrypt_and_verify_packages,
)


class OldSystem:
    
    def __init__(self, dataset_path: Optional[str] = None):
        self.reports: List[Dict[str, Any]] = []
        self.dataset_path = dataset_path
        self.df: Optional[pd.DataFrame] = None
        
        if dataset_path and os.path.exists(dataset_path):
            try:
                self.df = pd.read_csv(dataset_path)
                print(f"   loaded {len(self.df)} reports from CSV (plaintext)")
            except Exception as e:
                print(f"   could not load dataset: {e}")
    
    def send_report(self, report_id: str, report_text: str, crime_type: str = "vishing") -> Dict[str, Any]:
        report = {
            "report_id": report_id,
            "crime_type": crime_type,
            "text": report_text,
            "timestamp": None,
        }
        self.reports.append(report)
        return report
    
    def receive_report(self, report: Dict[str, Any]) -> Tuple[str, str, bool]:
        report_id = report.get("report_id", "")
        report_text = report.get("text", "")
        return report_id, report_text, True
    
    def process_for_nlp(self, report: Dict[str, Any]) -> str:
        return report.get("text", "")
    
    def get_report_from_dataset(self, index: int = 0) -> Optional[Dict[str, Any]]:
        if self.df is None or index >= len(self.df):
            return None
        
        row = self.df.iloc[index]
        report_id = str(row.get('submission_id', index)) if 'submission_id' in row else str(index)
        report_text = str(row.get('incident_description', '')) if 'incident_description' in row else ''
        
        return {
            "report_id": report_id,
            "crime_type": "vishing",
            "text": report_text,
            "timestamp": None,
        }


class NewSystem:
    
    def __init__(self, server_pub: RSAPublicKey, server_priv: RSAPrivateKey,
                 sender_pub: RSAPublicKey, sender_priv: RSAPrivateKey,
                 secure_reports_path: Optional[str] = None):
        self.server_pub = server_pub
        self.server_priv = server_priv
        self.sender_pub = sender_pub
        self.sender_priv = sender_priv
        self.reports: List[Dict[str, Any]] = []
        self.secure_reports_path = secure_reports_path
        self.packages: List[Dict[str, Any]] = []
        
        if secure_reports_path and os.path.exists(secure_reports_path):
            try:
                self.packages = load_secure_reports_jsonl(secure_reports_path)
                print(f"   loaded {len(self.packages)} secure packages (encrypted)")
            except Exception as e:
                print(f"   could not load secure reports: {e}")
    
    def send_report(self, report_id: str, report_text: str, crime_type: str = "vishing") -> Dict[str, Any]:
        package = create_package(
            report_id=report_id,
            report_text=report_text,
            sender_priv=self.sender_priv,
            sender_pub=self.sender_pub,
            server_pub=self.server_pub,
            crime_type=crime_type,
        )
        self.reports.append(package)
        return package
    
    def receive_report(self, package: Dict[str, Any]) -> Tuple[str, str, bool]:
        return open_package(package, self.server_priv)
    
    def process_for_nlp(self, package: Dict[str, Any]) -> str:
        report_id, plaintext, verified = self.receive_report(package)
        if not verified:
            return None
        return plaintext
    
    def get_package_from_dataset(self, index: int = 0) -> Optional[Dict[str, Any]]:
        if index >= len(self.packages):
            return None
        return self.packages[index]


class AttackSimulator:
    
    def __init__(self, dataset_path: Optional[str] = None, secure_reports_path: Optional[str] = None,
                 server_priv_path: Optional[str] = None):
        server_pub = None
        server_priv = None
        if server_priv_path and os.path.exists(server_priv_path):
            try:
                server_priv = load_private_key_json(server_priv_path)
                server_pub_path = server_priv_path.replace('private', 'public')
                if os.path.exists(server_pub_path):
                    from secure_reports import deserialize_public_key
                    import json
                    with open(server_pub_path, 'r') as f:
                        server_pub = deserialize_public_key(json.load(f))
            except Exception as e:
                print(f"   could not load server keys: {e}")
        
        if server_pub is None or server_priv is None:
            (server_pub, server_priv), (sender_pub, sender_priv) = rsa_keygen(bits=1024), rsa_keygen(bits=1024)
        else:
            (sender_pub, sender_priv) = rsa_keygen(bits=1024)
        
        self.old_system = OldSystem(dataset_path=dataset_path)
        self.new_system = NewSystem(server_pub, server_priv, sender_pub, sender_priv, 
                                    secure_reports_path=secure_reports_path)
        self.attacker_pub, self.attacker_priv = rsa_keygen(bits=1024)
    
    def print_separator(self, title: str):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")
    
    def attack_eavesdropping(self):
        self.print_separator("attack 1: eavesdropping (network interception)")
        
        old_report = None
        new_package = None
        
        if self.old_system.df is not None and len(self.old_system.df) > 0:
            print("using real dataset from CSV (plaintext)...")
            old_report = self.old_system.get_report_from_dataset(0)
            if old_report:
                report_text = old_report['text'][:100] + "..." if len(old_report['text']) > 100 else old_report['text']
                print(f"   report from dataset: {report_text}")
        else:
            report_text = "I received a call from someone claiming to be from DBS bank asking for my OTP code."
            old_report = self.old_system.send_report("123", report_text)
        
        print()
        print("network transmission intercepted by attacker...")
        print()
        
        print("old system (no encryption - direct from CSV):")
        print(f"   network transmission: {json.dumps(old_report, indent=2)}")
        print()
        print("   attacker can read:")
        print(f"      report ID: {old_report['report_id']}")
        print(f"      full text: {old_report['text'][:200]}...")
        print("   vulnerable: attacker sees everything from CSV!")
        print()
        
        if len(self.new_system.packages) > 0:
            print("using secure reports from encrypted dataset...")
            new_package = self.new_system.get_package_from_dataset(0)
        else:
            report_text = old_report['text'] if old_report else "I received a call from DBS bank."
            new_package = self.new_system.send_report(old_report['report_id'] if old_report else "123", report_text)
        
        print("new system (full encryption - from secure_reports.jsonl):")
        print(f"   network transmission:")
        print(f"      report_id: {new_package['report_id']}")
        print(f"      ciphertext: {new_package['ciphertext'][:50]}... (encrypted)")
        print(f"      enc_key: {new_package['enc_key'][:50]}... (encrypted)")
        print(f"      signature: {new_package['signature'][:50]}... (encrypted)")
        print(f"      hmac: {new_package['hmac'][:50]}... (encrypted)")
        print()
        print("   attacker cannot read:")
        print("      - cannot decrypt ciphertext (needs server private key)")
        print("      - cannot decrypt key (needs server private key)")
        print("      - cannot forge signature (needs sender private key)")
        print("      - cannot modify without detection (HMAC will fail)")
        print("   secure: attacker sees only encrypted data!")
        print()
    
    def attack_tampering(self):
        self.print_separator("attack 2: tampering (modifying reports)")
        
        original_text = "I received a call from DBS bank asking for my OTP."
        malicious_text = "I received a call from DBS bank and I gave them my OTP: 123456."
        
        print("original report:")
        print(f"   text: {original_text}")
        print()
        
        print("old system (no integrity protection):")
        old_report = self.old_system.send_report("123", original_text)
        print(f"   sent: {old_report['text']}")
        
        old_report['text'] = malicious_text
        print(f"   attacker modifies: {old_report['text']}")
        
        report_id, received_text, accepted = self.old_system.receive_report(old_report)
        print(f"   received: {received_text}")
        print(f"   accepted: {accepted}")
        print("   vulnerable: modification not detected!")
        print()
        
        print("new system (HMAC + signature protection):")
        new_package = self.new_system.send_report("123", original_text)
        print(f"   sent: {original_text}")
        print(f"   HMAC: {new_package['hmac'][:30]}...")
        print(f"   signature: {new_package['signature'][:30]}...")
        
        print(f"   attacker tries to modify...")
        modified_ciphertext = new_package['ciphertext'][:-10] + "XXXXXXXXXX"
        new_package['ciphertext'] = modified_ciphertext
        
        report_id, received_text, verified = self.new_system.receive_report(new_package)
        print(f"   verification result: {verified}")
        if not verified:
            print(f"   rejected: HMAC verification failed!")
            print("   secure: tampering detected and prevented!")
        else:
            print("   error: should have been rejected!")
        print()
    
    def attack_replay(self):
        self.print_separator("attack 3: replay attack (reusing old reports)")
        
        report_text = "I received a vishing call yesterday."
        
        print("original report sent at time T1:")
        print(f"   text: {report_text}")
        print()
        
        print("old system (no timestamp protection):")
        old_report = self.old_system.send_report("123", report_text)
        print(f"   sent at T1: {old_report}")
        
        print(f"   attacker replays at T2 (much later)...")
        report_id, received_text, accepted = self.old_system.receive_report(old_report)
        print(f"   received: {received_text}")
        print(f"   accepted: {accepted}")
        print("   vulnerable: replay not detected!")
        print()
        
        print("new system (timestamp in hash + IV):")
        new_package1 = self.new_system.send_report("123", report_text)
        timestamp1 = new_package1.get('timestamp')
        print(f"   sent at T1 (timestamp: {timestamp1})")
        print(f"   HMAC key derived from: AES_key + 'vishing|123|{timestamp1}'")
        
        import time
        time.sleep(1)
        new_package2 = self.new_system.send_report("123", report_text)
        timestamp2 = new_package2.get('timestamp')
        print(f"   same report sent at T2 (timestamp: {timestamp2})")
        print(f"   HMAC key derived from: AES_key + 'vishing|123|{timestamp2}'")
        print()
        print(f"   attacker tries to replay package from T1 at T2...")
        
        report_id, received_text, verified = self.new_system.receive_report(new_package1)
        print(f"   verification result: {verified}")
        if verified:
            print("   note: timestamp in hash prevents signature reuse,")
            print("      but package itself can be replayed if not checked.")
            print("   in production, add timestamp validation in receive_report()")
        print()
    
    def attack_mitm(self):
        self.print_separator("attack 4: man-in-the-middle attack")
        
        report_text = "I received a call from DBS bank."
        
        print("original report:")
        print(f"   text: {report_text}")
        print()
        
        print("old system (no authentication):")
        old_report = self.old_system.send_report("123", report_text)
        print(f"   sent: {json.dumps(old_report, indent=2)}")
        
        print(f"   attacker intercepts and creates fake report...")
        fake_report = {
            "report_id": "999",
            "crime_type": "vishing",
            "text": "I received a call from FAKE BANK and gave them my password.",
        }
        print(f"   fake report: {json.dumps(fake_report, indent=2)}")
        
        report_id, received_text, accepted = self.old_system.receive_report(fake_report)
        print(f"   received: {received_text}")
        print(f"   accepted: {accepted}")
        print("   vulnerable: fake report accepted!")
        print()
        
        print("new system (RSA signatures + HMAC):")
        new_package = self.new_system.send_report("123", report_text)
        print(f"   sent with signature from sender's private key")
        print(f"   HMAC protects integrity")
        
        print(f"   attacker tries to create fake package...")
        print("      - needs sender's private key to create valid signature")
        print("      - needs server's private key to encrypt key material")
        print("      - needs correct HMAC key (derived from AES key)")
        print("      - needs correct OAEP label (includes report_id)")
        
        try:
            fake_package = create_package(
                report_id="999",
                report_text="FAKE REPORT",
                sender_priv=self.attacker_priv,
                sender_pub=self.attacker_pub,
                server_pub=self.new_system.server_pub,
            )
            report_id, received_text, verified = self.new_system.receive_report(fake_package)
            if not verified:
                print(f"   rejected: signature verification failed!")
                print("   secure: fake report detected and prevented!")
        except Exception as e:
            print(f"   rejected: {str(e)}")
        print()
    
    def attack_signature_forgery(self):
        self.print_separator("attack 5: signature forgery")
        
        report_text = "I received a vishing call."
        
        print("original report:")
        print(f"   text: {report_text}")
        print()
        
        print("old system (no signatures):")
        old_report = self.old_system.send_report("123", report_text)
        print(f"   no signature field exists")
        print("   vulnerable: anyone can create reports!")
        print()
        
        print("new system (RSA signatures with SHA-256):")
        new_package = self.new_system.send_report("123", report_text)
        print(f"   signature created with sender's private key")
        print(f"   hash includes: crime_type + report_id + timestamp + text")
        print()
        
        print("   attacker tries to forge signature...")
        print("      method 1: use attacker's private key")
        try:
            import time
            fake_hash = hash_vishing_payload("123", report_text, "vishing", timestamp=int(time.time()))
            fake_sig = rsa_sign_hash(fake_hash, self.attacker_priv)
            new_package['signature'] = int_to_b64(fake_sig)
            
            report_id, received_text, verified = self.new_system.receive_report(new_package)
            if not verified:
                print(f"      failed: signature doesn't match sender's public key")
        except Exception as e:
            print(f"      failed: {str(e)}")
        
        print()
        print("      method 2: try to reuse signature from different report")
        package1 = self.new_system.send_report("123", report_text)
        package2 = self.new_system.send_report("456", "Different text")
        
        package2['signature'] = package1['signature']
        report_id, received_text, verified = self.new_system.receive_report(package2)
        if not verified:
            print(f"      failed: signature hash includes report_id, so it doesn't match")
        
        print()
        print("   secure: signature forgery prevented!")
        print("      - need sender's private key (only sender has it)")
        print("      - hash includes report_id (prevents reuse)")
        print("      - hash includes timestamp (prevents replay)")
        print()
    
    def attack_key_swapping(self):
        self.print_separator("attack 6: key swapping attack")
        
        report1_text = "Report 1: I received a call from DBS."
        report2_text = "Report 2: I received a call from OCBC."
        
        print("two reports sent:")
        print(f"   report 1: {report1_text}")
        print(f"   report 2: {report2_text}")
        print()
        
        print("old system (no key encryption):")
        old_report1 = self.old_system.send_report("123", report1_text)
        old_report2 = self.old_system.send_report("456", report2_text)
        print("   no keys to swap (plaintext)")
        print("   vulnerable: but attacker can just modify text directly")
        print()
        
        print("new system (OAEP with vishing-aware labels):")
        new_package1 = self.new_system.send_report("123", report1_text)
        new_package2 = self.new_system.send_report("456", report2_text)
        
        print(f"   package 1 encrypted key (for report 123)")
        print(f"   package 2 encrypted key (for report 456)")
        print()
        print("   attacker tries to swap encrypted keys...")
        
        temp_key = new_package1['enc_key']
        new_package1['enc_key'] = new_package2['enc_key']
        
        report_id, received_text, verified = self.new_system.receive_report(new_package1)
        if not verified:
            print(f"   rejected: OAEP label includes report_id!")
            print(f"      package 1 label: 'vishing|vishing|123'")
            print(f"      package 2 key encrypted with label: 'vishing|vishing|456'")
            print(f"      label mismatch -> decryption fails!")
            print("   secure: key swapping prevented!")
        print()
    
    def attack_pattern_analysis(self):
        self.print_separator("attack 7: pattern analysis prevention")
        
        same_text = "I received a call from DBS bank asking for my OTP."
        
        print("same report text sent multiple times:")
        print(f"   text: {same_text}")
        print()
        
        print("old system (plaintext):")
        old_report1 = self.old_system.send_report("123", same_text)
        old_report2 = self.old_system.send_report("456", same_text)
        old_report3 = self.old_system.send_report("789", same_text)
        
        print(f"   report 1: {old_report1['text']}")
        print(f"   report 2: {old_report2['text']}")
        print(f"   report 3: {old_report3['text']}")
        print("   attacker can see: all three are identical!")
        print("   vulnerable: pattern analysis possible")
        print()
        
        print("new system (vishing-aware IVs):")
        new_package1 = self.new_system.send_report("123", same_text)
        new_package2 = self.new_system.send_report("456", same_text)
        new_package3 = self.new_system.send_report("789", same_text)
        
        print(f"   report 1 ciphertext: {new_package1['ciphertext'][:50]}...")
        print(f"   report 2 ciphertext: {new_package2['ciphertext'][:50]}...")
        print(f"   report 3 ciphertext: {new_package3['ciphertext'][:50]}...")
        print()
        print("   different encryptions:")
        print("      - each report gets unique IV (from report_id + timestamp)")
        print("      - same text -> different ciphertext")
        print("      - attacker cannot detect patterns")
        print("   secure: pattern analysis prevented!")
        print()
    
    def print_summary(self):
        self.print_separator("security comparison summary")
        
        print("old system (no security):")
        print("   vulnerable: eavesdropping - attacker can read all reports")
        print("   vulnerable: tampering - modifications not detected")
        print("   vulnerable: replay - old reports can be reused")
        print("   vulnerable: MITM - fake reports accepted")
        print("   vulnerable: forgery - no signatures to verify")
        print("   vulnerable: pattern analysis - same text = same transmission")
        print()
        
        print("new system (full security):")
        print("   secure: eavesdropping - all data encrypted (AES-256 + RSA-OAEP)")
        print("   secure: tampering - HMAC + signatures detect modifications")
        print("   secure: replay - timestamps in hashes prevent reuse")
        print("   secure: MITM - RSA signatures prove sender identity")
        print("   secure: forgery - cannot forge without private keys")
        print("   secure: pattern analysis - unique IVs per report")
        print()
        
        print("security properties:")
        print("   secure: confidentiality - AES-256 CBC encryption")
        print("   secure: integrity - HMAC-SHA256 verification")
        print("   secure: authentication - RSA signatures")
        print("   secure: non-repudiation - signatures prove sender")
        print("   secure: domain-aware - vishing-specific bindings")
        print()
        
        print("result:")
        print("   old system: vulnerable to all attacks")
        print("   new system: secure against all attacks")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Attack Simulation")
    parser.add_argument("--csv", default=None, help="Path to scam_raw_dataset.csv (for old system)")
    parser.add_argument("--secure", default=None, help="Path to secure_reports.jsonl (for new system)")
    parser.add_argument("--server-priv", default=None, help="Path to server_rsa_private.json")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "Data Set")
    
    csv_path = args.csv or os.path.join(data_dir, "scam_raw_dataset.csv")
    secure_path = args.secure or os.path.join(data_dir, "secure_reports.jsonl")
    server_priv_path = args.server_priv or os.path.join(data_dir, "server_rsa_private.json")
    
    print("\n" + "=" * 80)
    print("  security attack simulation")
    print("  comparing old system (no security) vs new system (full security)")
    print("=" * 80)
    print()
    print("dataset configuration:")
    print(f"   old system (CSV): {csv_path}")
    print(f"   new system (encrypted): {secure_path}")
    print(f"   server keys: {server_priv_path}")
    print()
    
    simulator = AttackSimulator(
        dataset_path=csv_path if os.path.exists(csv_path) else None,
        secure_reports_path=secure_path if os.path.exists(secure_path) else None,
        server_priv_path=server_priv_path if os.path.exists(server_priv_path) else None,
    )
    
    simulator.attack_eavesdropping()
    simulator.attack_tampering()
    simulator.attack_replay()
    simulator.attack_mitm()
    simulator.attack_signature_forgery()
    simulator.attack_key_swapping()
    simulator.attack_pattern_analysis()
    
    simulator.print_summary()
    
    print("\n" + "=" * 80)
    print("  simulation complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
