import json

def team_gold(players, is_radiant):
    gold = 0
    for i in players:
        if(i['isRadiant'] == is_radiant):
            gold = gold + i['total_gold']
    
    return int(gold/5)

def team_xp(players, is_radiant):
    xp = 0
    for i in players:
        if(i['isRadiant'] == is_radiant):
            xp = xp + i['total_xp']
    
    return int(xp/5)


with open('matches.txt') as f:
    with open('dataset.txt', 'w') as d:
        d.write("%SCORE\tGOLD\tXP\tRESULT\n")
        parsed = json.load(f)
        for i in parsed:
            match = {}
            #All features are described as the difference
            try:
                match['score'] = i['radiant_score'] - i['dire_score']
                match['gold'] = team_gold(i['players'], True) - team_gold(i['players'], False)
                match['xp'] = team_xp(i['players'], True) - team_xp(i['players'], False)
                match['result'] = 1 if i['radiant_win'] else 0
                d.write(str(match['score'])  
                    + "\t" +str(match['gold'])
                    + "\t" + str(match['xp']) 
                    +  "\t" + str(match['result']) + "\n")
            except:
                continue
print('END')
