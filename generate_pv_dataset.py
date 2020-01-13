import requests
import csv


# Generate training and evaluation dataset for model
# Send http requests to techslides.com to retreive latitude and longitude of all capitals in the world
# Send http requests to solcast.com to retreive weather and solar radiation measurements and corresponding PV power estimates

dataset = []
countriesResponse = requests.get('http://techslides.com/demos/country-capitals.json').json()

for country in countriesResponse:
    tmpDataset = []
    
    worldRadiationEstimates = requests.get('https://api.solcast.com.au/world_radiation/estimated_actuals?api_key=WzoGxTP-4ikpr7vYtKKg73FDUl0AoYmJ&latitude=' + country['CapitalLatitude'] + '&longitude=' + country['CapitalLongitude'] + '&format=json&hours=168').json()
    worldPVPowerEstimates = requests.get('https://api.solcast.com.au/world_pv_power/estimated_actuals?api_key=WzoGxTP-4ikpr7vYtKKg73FDUl0AoYmJ&latitude=' + country['CapitalLatitude'] + '&longitude=' + country['CapitalLongitude'] + '&loss_factor=0.9&capacity=10&tilt=23&azimuth=0&hours=168&format=json').json()
    
    if 'estimated_actuals' not in worldRadiationEstimates or 'estimated_actuals' not in worldPVPowerEstimates:
        continue
        
    worldRadiationEstimates = worldRadiationEstimates['estimated_actuals']
    worldPVPowerEstimates = worldPVPowerEstimates['estimated_actuals']
    
    for entry in worldRadiationEstimates:
        del entry['period_end']
        del entry['period']
        
    for (radiationEntry, powerEntry) in zip(worldRadiationEstimates, worldPVPowerEstimates):
    
        tmpDataset.append({**radiationEntry, **{'capacity': 10}, **powerEntry})
        
    dataset += tmpDataset
    
with open('PV_dataset.csv', 'w') as datasetCsv:
    writer = csv.DictWriter(datasetCsv, fieldnames = [*dataset[0]])
    writer.writeheader()
    
    for entry in dataset:
        writer.writerow(entry)
        
datasetCsv.close()

    
