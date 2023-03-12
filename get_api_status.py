import requests
from bs4 import BeautifulSoup
import streamlit as st

@st.cache_data
def get_api_status():
    response = requests.get("https://status.openai.com/")
    if response.ok:
        html = response.content
        soup = BeautifulSoup(html, "html.parser")
        name = soup.select_one(
            "body > div.layout-content.status.status-index.starter > div.container > div.components-section.font-regular > div.components-container.one-column > div:nth-child(1) > div > span.name"
        )
        if name.text.strip() == "API":
            element = soup.select_one(
                "body > div.layout-content.status.status-index.starter > div.container > div.components-section.font-regular > div.components-container.one-column > div:nth-child(1) > div > span.component-status"
            )
            content = element.text.strip()
            if content == "Operational":
                return ":green[Operational]"
            else:
                return content
        else:
            return "Failed to get. Check on https://status.openai.com/"
    else:
        return f"Response status code: {response.status_code}"
