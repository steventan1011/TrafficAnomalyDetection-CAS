from OSMPythonTools.api import Api
from OSMPythonTools.overpass import Overpass
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import overpassQueryBuilder, Overpass

if __name__ == '__main__':

    api = Api()
    way = api.query('way/5887599')
    print(way.tag('longitude'))
    print(way.tag('latitude'))
    print(way.tag('building'))
    # 'castle'
    print(way.tag('architect'))
    # 'Johann Lucas von Hildebrandt'
    print(way.tag('website'))
    # 'http://www.belvedere.at'

    overpass = Overpass()
    result = overpass.query('way["name"="Stephansdom"]; out body;')
    stephansdom = result.elements()[0]
    stephansdom.tag('name:en')
    # "Saint Stephen's Cathedral"
    '%s %s, %s %s' % (
    stephansdom.tag('addr:street'), stephansdom.tag('addr:housenumber'), stephansdom.tag('addr:postcode'),
    stephansdom.tag('addr:city'))
    # 'Stephansplatz 3, 1010 Wien'
    stephansdom.tag('building')
    # 'cathedral'
    stephansdom.tag('denomination')
    # 'catholic'、

    nominatim = Nominatim()
    areaId = nominatim.query('Beijing, China').areaId()
    print(areaId)

    overpass = Overpass()
    query = overpassQueryBuilder(area=areaId, elementType='node', selector='"natural"="tree"', out='count')
    result = overpass.query(query)
    print(result.countElements())