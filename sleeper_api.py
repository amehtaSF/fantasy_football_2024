
import requests
import json
import os
from pprint import pprint


class SleeperAPI:
    
    def __init__(self, league_id=None):
        self.base_url = 'https://api.sleeper.app/v1/'
        
        self.league_id = league_id
        self.league = None
        self.draft = None
        self.draft_picks = None
        
    def get_users_in_league(self, league_id=None):
        assert league_id is not None or self.league_id is not None, 'League ID must be provided'
        league_id = league_id if league_id is not None else self.league_id
        url = self.base_url + 'league/' + str(self.league_id) + '/users'
        response = requests.get(url)
        users = response.json()
        self.league = users
        return users
    
    def get_username(self, user_id):
        league = self.league if self.league is not None else self.get_users_in_league()
        return [user['display_name'] for user in league if user['user_id'] == user_id][0]
    
    def get_draft_in_league(self, league_id=None):
        assert league_id is not None or self.league_id is not None, 'League ID must be provided'
        league_id = league_id if league_id is not None else self.league_id
        url = self.base_url + 'league/' + str(self.league_id) + '/drafts'
        response = requests.get(url)
        drafts = response.json()
        if len(drafts) > 1:
            raise RuntimeWarning(f"{len(drafts)} found drafts found. Returning first draft only.")
        self.draft = drafts[0]
        return drafts[0]
    
    def get_draft_picks(self, draft_id=None):
        assert draft_id is not None or self.draft is not None
        draft_id = draft_id if draft_id is not None else self.draft["draft_id"]
        url = self.base_url + 'draft/' + draft_id + '/picks'
        response = requests.get(url)
        picks = response.json()
        self.draft_picks = picks
        return picks
    

if __name__ == '__main__':
    
    LEAGUE_ID = 1119159922530312192
    sleeper = SleeperAPI(LEAGUE_ID)
    # users = sleeper.get_users_in_league()
    # pprint(users)
    draft = sleeper.get_draft_in_league()
    # pprint(draft)
    pprint(sleeper.get_draft_picks(draft["draft_id"]))