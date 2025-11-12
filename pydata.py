from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
parkinsons = fetch_ucirepo(id=174) 
  
# data (as pandas dataframes) 
X = parkinsons.data.features 
y = parkinsons.data.targets 
print(X)
print(y)
# metadata 
print(parkinsons.metadata) 
  
# variable information 
print(parkinsons.variables) 
