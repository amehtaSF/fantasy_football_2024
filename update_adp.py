import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yaml
import pandas as pd

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


SLEEPER_ADP_SHEET = config["urls"]["sleeper"]["adp_gsheet"]
gspread_key = config["keys"]["gspread"]
# Define the scope and credentials
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name(gspread_key, scope)


if __name__ == "__main__":
    
    # Authenticate and open the Google Sheet
    gc = gspread.authorize(credentials)

    sheet = gc.open_by_url(SLEEPER_ADP_SHEET)
    vals = sheet.get_worksheet(1).get_all_values()
    
    df_adp = pd.DataFrame(vals[1:], columns=vals[0])
    
    

    df_adp.to_csv(config["files"]["sleeper"]["adp"], index=False)
