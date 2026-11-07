from pathlib import Path
import trafilatura

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENTS = [
    # Microsoft Learn
    {
        "id": "DOC001",
        "filename": "windows_dns_troubleshooting.md",
        "title": "Résolution des problèmes liés aux clients DNS",
        "url": "https://learn.microsoft.com/fr-fr/windows-server/networking/dns/troubleshoot/troubleshoot-dns-client",
    },
    {
        "id": "DOC002",
        "filename": "windows_ipconfig.md",
        "title": "Commande ipconfig",
        "url": "https://learn.microsoft.com/fr-fr/windows-server/administration/windows-commands/ipconfig",
    },
    {
        "id": "DOC003",
        "filename": "windows_dns_client_settings.md",
        "title": "Bonnes pratiques des paramètres du client DNS",
        "url": "https://learn.microsoft.com/fr-fr/troubleshoot/windows-server/networking/best-practices-for-dns-client-settings",
    },

    # Ubuntu
    {
        "id": "DOC004",
        "filename": "ubuntu_netplan.md",
        "title": "Configuration réseau avec Netplan",
        "url": "https://netplan.readthedocs.io/en/stable/",
    },
    {
        "id": "DOC005",
        "filename": "ubuntu_dns.md",
        "title": "Domain Name Service (DNS)",
        "url": "https://ubuntu.com/server/docs/how-to/networking/install-dns/",
    },
    {
        "id": "DOC006",
        "filename": "ubuntu_networking.md",
        "title": "Documentation réseau Ubuntu",
        "url": "https://ubuntu.com/server/docs/how-to/networking/",
    },

    # OpenVPN
    {
        "id": "DOC007",
        "filename": "openvpn_installation.md",
        "title": "Installing OpenVPN",
        "url": "https://openvpn.net/community-docs/installing-openvpn/",
    },
    {
        "id": "DOC008",
        "filename": "openvpn_connectivity.md",
        "title": "Starting Up the VPN and Testing Connectivity",
        "url": "https://openvpn.net/community-docs/starting-up-the-vpn-and-testing-for-initial-connectivity/",
    },
    {
        "id": "DOC009",
        "filename": "openvpn_getting_started.md",
        "title": "Getting Started with OpenVPN",
        "url": "https://openvpn.net/community-docs/how-to/",
    },
]


def download_document(document):
    print(f"\nTéléchargement : {document['id']}")

    downloaded = trafilatura.fetch_url(document["url"])

    if not downloaded:
        print(f"ERREUR : impossible de récupérer {document['url']}")
        return False

    content = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_links=True,
        include_tables=True,
    )

    if not content or len(content.strip()) < 100:
        print("ERREUR : contenu extrait insuffisant.")
        return False

    header = f"""---
doc_id: {document['id']}
title: "{document['title']}"
source_url: "{document['url']}"
---

"""

    output_path = OUTPUT_DIR / document["filename"]
    output_path.write_text(
        header + content,
        encoding="utf-8"
    )

    print(f"OK : {output_path}")
    print(f"Nombre de caractères : {len(content)}")

    return True


if __name__ == "__main__":
    success = 0

    for document in DOCUMENTS:
        if download_document(document):
            success += 1

    print("\n--------------------------------")
    print(f"Documents récupérés : {success}/{len(DOCUMENTS)}")
    print("--------------------------------")