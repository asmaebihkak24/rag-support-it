
import csv
import hashlib
from datetime import date
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/sources.csv")

DOCUMENTS = [
    {
        "doc_id": "DOC001",
        "filename": "windows_dns_troubleshooting.md",
        "title": "Résolution des problèmes DNS côté client",
        "organization": "Microsoft",
        "os_scope": "Windows",
        "url": "https://learn.microsoft.com/fr-fr/windows-server/networking/dns/troubleshoot/troubleshoot-dns-client",
    },
    {
        "doc_id": "DOC002",
        "filename": "windows_ipconfig.md",
        "title": "Commande ipconfig",
        "organization": "Microsoft",
        "os_scope": "Windows",
        "url": "https://learn.microsoft.com/fr-fr/windows-server/administration/windows-commands/ipconfig",
    },
    {
        "doc_id": "DOC003",
        "filename": "windows_dns_client_settings.md",
        "title": "Paramètres du client DNS",
        "organization": "Microsoft",
        "os_scope": "Windows",
        "url": "https://learn.microsoft.com/fr-fr/troubleshoot/windows-server/networking/best-practices-for-dns-client-settings",
    },
    {
        "doc_id": "DOC004",
        "filename": "ubuntu_netplan.md",
        "title": "Documentation Netplan",
        "organization": "Netplan",
        "os_scope": "Ubuntu/Linux",
        "url": "https://netplan.readthedocs.io/en/stable/",
    },
    {
        "doc_id": "DOC005",
        "filename": "ubuntu_dns.md",
        "title": "Domain Name Service (DNS)",
        "organization": "Ubuntu",
        "os_scope": "Ubuntu/Linux",
        "url": "https://ubuntu.com/server/docs/how-to/networking/install-dns/",
    },
    {
        "doc_id": "DOC006",
        "filename": "ubuntu_networking.md",
        "title": "Documentation réseau Ubuntu",
        "organization": "Ubuntu",
        "os_scope": "Ubuntu/Linux",
        "url": "https://ubuntu.com/server/docs/how-to/networking/",
    },
    {
        "doc_id": "DOC007",
        "filename": "openvpn_installation.md",
        "title": "Installing OpenVPN",
        "organization": "OpenVPN",
        "os_scope": "Windows/Linux",
        "url": "https://openvpn.net/community-docs/installing-openvpn/",
    },
    {
        "doc_id": "DOC008",
        "filename": "openvpn_connectivity.md",
        "title": "Starting Up the VPN and Testing Connectivity",
        "organization": "OpenVPN",
        "os_scope": "Windows/Linux",
        "url": "https://openvpn.net/community-docs/starting-up-the-vpn-and-testing-for-initial-connectivity/",
    },
    {
        "doc_id": "DOC009",
        "filename": "openvpn_getting_started.md",
        "title": "Getting Started with OpenVPN",
        "organization": "OpenVPN",
        "os_scope": "Windows/Linux",
        "url": "https://openvpn.net/community-docs/how-to/",
    },
]


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(4096), b""):
            sha256.update(block)

    return sha256.hexdigest()


def main():
    rows = []

    for document in DOCUMENTS:
        file_path = RAW_DIR / document["filename"]

        if not file_path.exists():
            raise FileNotFoundError(f"Document introuvable : {file_path}")

        if file_path.stat().st_size == 0:
            raise ValueError(f"Document vide : {file_path}")

        rows.append({
            "doc_id": document["doc_id"],
            "filename": document["filename"],
            "title": document["title"],
            "organization": document["organization"],
            "source_type": "Official documentation",
            "os_scope": document["os_scope"],
            "url": document["url"],
            "publication_date": "",
            "version": "",
            "download_date": date.today().isoformat(),
            "license": "À vérifier",
            "sha256": calculate_sha256(file_path),
            "notes": "",
        })

    fieldnames = [
        "doc_id", "filename", "title", "organization",
        "source_type", "os_scope", "url", "publication_date",
        "version", "download_date", "license", "sha256", "notes"
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Catalogue créé : {OUTPUT_FILE}")
    print(f"Nombre de documents : {len(rows)}")
    print("SHA256 calculé pour chaque document.")


if __name__ == "__main__":
    main()
