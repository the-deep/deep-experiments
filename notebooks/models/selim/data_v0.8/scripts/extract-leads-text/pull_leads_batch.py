import io
import json
import boto3
# import timeout_decorator
import pandas as pd

from deep_parser import TextFromFile, TextFromWeb
from deep_parser.helpers.errors import DocumentProcessingError

BUCKET_NAME: str = "pulled-leads"
DATAFRAME_BUCKET_PATH: str = "projects-leads-urls/projects_leads_urls.csv"
S3_RESOURCE = boto3.resource("s3", "us-east-1")

def get_projects_leads_dataset():

    obj = S3_RESOURCE.Object(
        BUCKET_NAME, 
        DATAFRAME_BUCKET_PATH
        )

    leads_df = pd.read_csv(
        io.StringIO(
            obj.get()['Body'].read().decode()
            )
        )

    return leads_df


def save_object(text: dict, project_id: int, lead_id: int):

    s3object = S3_RESOURCE.Object(BUCKET_NAME, f'{project_id}/{lead_id}.json')
    s3object.put(
        Body=(bytes(json.dumps(text).encode('UTF-8')))
    )


#@timeout_decorator.timeout(5 * 60, use_signals=False)
def pull_websites(leads_df: pd.DataFrame):

    print("###################### Start puling websites data.")
    
    for (project_id, lead_id), group in list(leads_df.groupby(["project_id", "lead_id"])):
        project_id, lead_id = str(int(project_id)), str(int(lead_id))
        urls = group["url"].unique()

        assert len(urls) == 1

        url = urls[0]

        if not isinstance(url, str):  # possibly np.nan
            continue

        if url.endswith(".pdf"):
            print(f"{url} is a PDF!")
            try:
                parser = TextFromFile(url=url, from_web=True)
                text, _ = parser.extract_text(output_format="list")
                save_object(text=text, project_id=project_id, lead_id=lead_id)
                print(f"project-id: {project_id}, lead-id: {lead_id}, type: pdf. correctly processed")
            except (RuntimeError, DocumentProcessingError, Exception) as e:
                print(f"Error {e} on PDF url {url}")
                continue
        else:
            try:
                parser = TextFromWeb(url=url)
                text = parser.extract_text(output_format="list")
                parser.close()
                save_object(text=text, project_id=project_id, lead_id=lead_id)
                print(f"project-id: {project_id}, lead-id: {lead_id}, type: website. correctly processed")
            except (RuntimeError, DocumentProcessingError, Exception) as e:
                print(f"Error {e} on standard url {url}")
                continue

if __name__ == "__main__":

    leads_df = get_projects_leads_dataset()
    pull_websites(leads_df=leads_df)
