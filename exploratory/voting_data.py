import requests, zipfile, io

url = "https://raw.github.com/mggg-states/NC-shapefiles/master/NC_VTD.zip"
r = requests.get(url, stream=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("data/nc")