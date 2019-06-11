from urllib.request import Request, urlopen
import json
import time

def GET(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    contents = urlopen(req).read().decode('utf-8')
    return contents

#Retrieves my matches
contents =  GET("https://api.opendota.com/api/players/214633926/matches")
jsonBody = json.loads(contents)
print("The number of the all matches " + str(len(jsonBody)))


#Loop through all of them to retrieve more info per match
allMatches = []
counter = 1
for i in jsonBody:
    print(str(counter)+". Retrieving match " + str(i['match_id']), end=' ')
    counter = counter + 1
    try:
        matchInfo = json.loads(
            GET("https://api.opendota.com/api/matches/" + str(i['match_id'])))
        print("S")
    except:
        print("F")
        continue
    allMatches.append(matchInfo) 
    time.sleep(8) #Because opendota is not generous :D and has call limit

with open('matches.txt', 'w') as f:
    f.write(json.dumps(allMatches))
   



