from pydantic import BaseModel, Field
from typing import List, Dict, Self
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time

feeds = [
    "https://www.mydealz.de/rss/alles",
    "https://www.mydealz.de/rssx/keyword-alarm/gScRn4Onx8DksO58d47q_vyHN6S3L79YItXsAImGzl8.",
]


def extract(html_snippet: str) -> str:
    """
    Use Beautiful Soup to clean up this HTML snippet and extract useful text
    """
    soup = BeautifulSoup(html_snippet, "html.parser")
    content = soup.find("div", class_="snippet summary") or soup

    text = content.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    return text


class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed
    """

    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        """
        Populate this instance based on the provided dict
        """
        self.title = entry.get("title", "No Title")
        self.url = entry.get("link", "")
        if "links" in entry and entry["links"]:
            self.url = entry["links"][0]["href"]

        summary_html = entry.get("summary", "")
        self.summary = extract(summary_html)

        # Handle MyDealz/Pepper specific fields
        self.price = ""
        self.merchant = ""

        if "pepper_merchant" in entry:
            merchant_info = entry["pepper_merchant"]
            self.merchant = merchant_info.get("name", "")
            self.price = merchant_info.get("price", "")
            if self.price:
                self.title = f"{self.title} ({self.price})"

        # Decide whether to fetch external content
        # MyDealz feeds usually have a good description in the RSS summary
        if "pepper_merchant" in entry or len(self.summary) > 200:
            self.details = self.summary
            self.features = ""
        else:
            try:
                stuff = requests.get(self.url, timeout=5).content
                soup = BeautifulSoup(stuff, "html.parser")
                content_div = soup.find("div", class_="content-section")
                if content_div:
                    content = content_div.get_text()
                    content = content.replace("\nmore", "").replace("\n", " ")
                    if "Features" in content:
                        self.details, self.features = content.split("Features", 1)
                    else:
                        self.details = content
                        self.features = ""
                else:
                    self.details = self.summary
                    self.features = ""
            except Exception:
                self.details = self.summary
                self.features = ""

        self.truncate()

    def truncate(self):
        """
        Limit the fields to a sensible length to avoid sending too much info to the model
        """
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self):
        """
        Return a string to describe this deal
        """
        return f"<{self.title}>"

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        """
        Retrieve all deals from the selected RSS feeds
        """
        deals = []
        feed_iter = tqdm(feeds) if show_progress else feeds
        for feed_url in feed_iter:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:20]:
                deals.append(cls(entry))
                time.sleep(0.05)
        return deals


class Deal(BaseModel):
    """
    A class to Represent a Deal with a summary description
    """

    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a short paragraph of text for each item you choose."
    )
    price: float = Field(
        description="The actual price of this product, as advertised in the deal. Be sure to give the actual price; for example, if a deal is described as $100 off the usual $300 price, you should respond with $200"
    )
    url: str = Field(description="The URL of the deal, as provided in the input")


class DealSelection(BaseModel):
    """
    A class to Represent a list of Deals
    """

    deals: List[Deal] = Field(
        description="Your selection of the 5 deals that have the most detailed, high quality description and the most clear price. You should be confident that the price reflects the deal, that it is a good deal, with a clear description"
    )


class Opportunity(BaseModel):
    """
    A class to represent a possible opportunity: a Deal where we estimate
    it should cost more than it's being offered
    """

    deal: Deal
    estimate: float
    discount: float
