"""Téléchargement des documents du corpus (version 2 : blocs de code conservés).

Avant l'extraction par trafilatura, le contenu de chaque bloc <pre> est aplati en
texte brut, pour que les commandes ne soient plus perdues.

Lancer depuis la racine du projet : python download_docs.py
"""

from pathlib import Path
import re
import lxml.html
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


def parse_html(html):
    try:
        return lxml.html.fromstring(html)
    except ValueError:
        parser = lxml.html.HTMLParser(encoding="utf-8")
        return lxml.html.fromstring(html.encode("utf-8"), parser=parser)


def has_class(element, name):
    return name in (element.get("class") or "").split()


def convert_terminal_blocks(tree):
    """Transforme les blocs <div class="terminal"> (Sphinx, bouton de copie) en vrais blocs de code.

    Structure observée sur la documentation Ubuntu :
    div.terminal > div.input > span.command (la commande) + prompt "$".
    trafilatura conserve l'invite mais perd la commande : on la réécrit en <pre><code>.
    """
    for block in list(tree.iter("div")):
        if not has_class(block, "terminal"):
            continue
        lines = []
        for part in block:
            if not isinstance(part.tag, str):
                continue
            if has_class(part, "input"):
                commands = [
                    el for el in part.iter()
                    if isinstance(el.tag, str) and has_class(el, "command")
                ]
                if commands:
                    lines.append("$ " + " ".join(commands[0].text_content().split()))
                else:
                    lines.append(" ".join(part.text_content().split()))
            else:
                lines.extend(
                    line.rstrip() for line in part.text_content().strip("\n").splitlines()
                )
        if not any(lines):
            continue
        pre = lxml.html.Element("pre")
        code = lxml.html.Element("code")
        pre.append(code)
        code.text = "\n".join(lines)
        pre.tail = block.tail
        block.getparent().replace(block, pre)


def flatten_code_blocks(html):
    """Prépare le HTML : blocs terminal convertis, puis chaque <pre> aplati en texte brut."""
    tree = parse_html(html)
    convert_terminal_blocks(tree)
    for pre in list(tree.iter("pre")):
        text = pre.text_content()
        for child in list(pre):
            pre.remove(child)
        pre.text = text
    return lxml.html.tostring(tree, encoding="unicode")


def download_document(document):
    print(f"\nTéléchargement : {document['id']}")

    downloaded = trafilatura.fetch_url(document["url"])

    if not downloaded:
        print(f"ERREUR : impossible de récupérer {document['url']}")
        return False

    html = flatten_code_blocks(downloaded)

    content = trafilatura.extract(
        html,
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
    output_path.write_text(header + content, encoding="utf-8")

    # Contrôle qualité : des invites de commande sans commande signalent une perte de code
    prompts = len(re.findall(r"(?m)^\s*\$\s+\S", content))
    code_fences = content.count("```") // 2
    sudo_count = content.count("sudo")
    print(f"OK : {output_path}")
    print(f"Caractères : {len(content)} | blocs de code : {code_fences} | invites $ : {prompts} | sudo : {sudo_count}")
    if prompts and not sudo_count:
        print("ATTENTION : des invites $ sont présentes mais aucune commande sudo. Vérifiez le fichier.")

    return True


if __name__ == "__main__":
    success = 0

    for document in DOCUMENTS:
        if download_document(document):
            success += 1

    print("\n--------------------------------")
    print(f"Documents récupérés : {success}/{len(DOCUMENTS)}")
    print("--------------------------------")