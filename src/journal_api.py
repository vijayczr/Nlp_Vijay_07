import requests
import pandas as pd

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"


def search_papers(
    query=None,
    venue=None,
    fields_of_study=None,
    publication_types=None,
    date_range=None,
    sort="publicationDate"
):
    params = {
        "fields": "paperId,title,year,abstract,venue,publicationDate",
        "sort": sort
    }

    if query:
        params["query"] = query

    if venue:
        params["venue"] = venue

    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study

    if publication_types:
        params["publicationTypes"] = publication_types

    if date_range:
        params["publicationDateOrYear"] = date_range

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    return response.json()


def get_papers_dataframe(
    query=None,
    venue=None,
    fields_of_study=None,
    publication_types=None,
    date_range=None,
    sort="publicationDate"
):
    result = search_papers(
        query=query,
        venue=venue,
        fields_of_study=fields_of_study,
        publication_types=publication_types,
        date_range=date_range,
        sort=sort
    )

    papers = result["data"]
    df = pd.DataFrame(papers)

    if "publicationDate" in df.columns:
        df["publicationDate"] = pd.to_datetime(df["publicationDate"], errors="coerce")
        df["month"] = df["publicationDate"].dt.month.fillna(1).astype(int)
        df["four_month_section"] = ((df["month"] - 1) // 4 + 1).astype(int)

    return df