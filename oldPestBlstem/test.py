import geopandas as gpd
path = r"F:\project\gitprojects\vscode\zsl\lxy\pestBIstm\datas\shandong_pest_data\shandong.json"
gdf = gpd.read_file(path)
print(len(gdf), gdf.columns)