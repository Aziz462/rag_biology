import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError
from lxml import etree
import os


class S3WorkerParser:
    def __init__(self):
        self.BUCKET_NAME = "pmc-oa-opendata"
        self.s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        
    def checkForRelevance(self, node):
        """
        Определяет, нужно ли сохранять секцию по ключевым словам.
        """
        tag_name = etree.QName(node).localname.lower()

        if tag_name == "abstract":
            return True

        sec_type = node.attrib.get("sec-type", "").lower() if hasattr(node, "attrib") else ""
        
        title_nodes = node.xpath("./*[local-name()='title']")
        title_text = title_nodes[0].xpath("string(.)").strip().lower() if title_nodes else ""

        if sec_type in ["intro", "discussion", "conclusions"]:
            return True

        keep_keywords = ["introduction", "background", "discussion", "conclusion", "summary"]
        if any(k in title_text for k in keep_keywords):
            return True

        return False


    def clean_text(self, node):
        text = "".join(node.itertext())
        return " ".join(text.split())


    def sectionToMarkdown(self, elem, level=2):
        if not self.checkForRelevance(elem):
            return ""
        
        md_output = ""

        title_node = elem.xpath("./*[local-name()='title']")
        
        if title_node:
            title_text = self.clean_text(title_node[0])
        else:
            title_text = "Untitled Section"
        
        md_output += f"{'#' * level} {title_text}\n\n"
        
        paragraphs = elem.xpath("./*[local-name()='p']")
        for p in paragraphs:
            text = self.clean_text(p)
            if text:
                md_output += f"{text}\n\n"

        subsections = elem.xpath("./*[local-name()='sec']")
        for sub in subsections:
            md_output += self.sectionToMarkdown(sub, level + 1)

        return md_output


    def xmlToMarkdown(self, root, pmcid):
        md_content = f"# PMC ID = {pmcid}"
        abstract_nodes = root.xpath(".//*[local-name()='abstract']")
        if abstract_nodes:
            md_content += "## Abstract\n\n"
            for ab in abstract_nodes:
                for child in ab.iterchildren():
                    if etree.QName(child).localname == 'p':
                        md_content += self.clean_text(child) + "\n\n"
                    elif etree.QName(child).localname == 'sec':
                        md_content += self.sectionToMarkdown(child, level=3)

        body_nodes = root.xpath(".//*[local-name()='body']")
        if body_nodes:
            sections = body_nodes[0].xpath("./*[local-name()='sec']")
            for sec in sections:
                md_content += self.sectionToMarkdown(sec, level=2)

        return md_content


    def printSectionsWithAbstract(self, root):
        """
        Вывод секций JATS XML, сначала абстракт, потом секции body
        """
        abstract_nodes = root.xpath(".//*[local-name()='abstract']")
        if abstract_nodes:
            for i, ab in enumerate(abstract_nodes, start=0):
                if self.checkForRelevance(ab):
                    print(f"\033[92m{i} | type='abstract' | title='Abstract'\033[0m")
                else:
                    print(f"\033[91m{i} | type='abstract' | title='Abstract'\033[0m")
        body_nodes = root.xpath(".//*[local-name()='body']")
        if body_nodes:
            self.printSections(body_nodes[0])


    def printSections(self, elem, prefix=None):
        """
        Вывод секций body JATS XML
        """
        if prefix is None:
            prefix = []

        sections = elem.xpath("./*[local-name()='sec']")
        
        for i, sec in enumerate(sections, start=1):
            number = prefix + [i]
            number_str = ".".join(map(str, number))

            title_node = sec.xpath("./*[local-name()='title']")
            
            if title_node:
                title_text = title_node[0].xpath("string(.)").strip()
            else:
                title_text = "[No Title]"

            sec_type = sec.attrib.get("sec-type", "")
            if self.checkForRelevance(sec):
                print(f"\033[92m{number_str} | type='{sec_type}' | title='{title_text}'\033[0m")
            else:
                print(f"\033[91m{number_str} | type='{sec_type}' | title='{title_text}'\033[0m")

            self.printSections(sec, number)

    def downloadSingleID(self, pmcid):
        if not os.path.exists("output"):
            os.makedirs("output")


        possibleLocations = [
        f"oa_comm/xml/all/{pmcid}.xml",
        f"oa_noncomm/xml/all/{pmcid}.xml",
        f"author_manuscript/xml/all/{pmcid}.xml"
        ]

        print(f"Looking for \"{pmcid}\"")

        xmlFile = f"{pmcid}.xml"
        
        for key in possibleLocations:
            try:
                self.s3.head_object(Bucket=self.BUCKET_NAME, Key=key)
                self.s3.download_file(self.BUCKET_NAME, key, xmlFile)
                tree = etree.parse(xmlFile)
                root = tree.getroot()
                self.printSectionsWithAbstract(root)
                with open(f"output/{pmcid}.md", "w") as f:
                    f.write(self.xmlToMarkdown(root, pmcid))
                os.remove(xmlFile)
                return True
            except ClientError:
                continue
        return False


    def downloadByIDs(self, pathToIds):
        if not os.path.exists("output"):
            os.makedirs("output")

        with open(pathToIds, "r") as f:
            for line in f.readlines():
                pmcid = line.strip()
                self.downloadSingleID(pmcid)
