# pgeocode

[![pypi](https://img.shields.io/pypi/v/pgeocode.svg)](https://pypi.org/project/pgeocode/)
[![rdfd](https://readthedocs.org/projects/pgeocode/badge/?version=latest)](http://pgeocode.readthedocs.io/)
[![GHactions](https://github.com/symerio/pgeocode/workflows/Test/badge.svg)](https://github.com/symerio/pgeocode/actions?query=branch%3Amaster+)

Postal code geocoding and distance calculations

pgeocode is a Python library for high performance off-line querying of
GPS coordinates, region name and municipality name from postal codes.
Distances between postal codes as well as general distance queries are
also supported. The used
[GeoNames](http://download.geonames.org/export/zip/) database includes
postal codes for 83 countries.

Currently, only queries within the same country are supported.

For additional documentation see
[pgeocode.readthedocs.io](https://pgeocode.readthedocs.io).

# Installation

pgeocode requires Python 3.10+ as well as `numpy` and `pandas` packages.
It can be installed with,

```
pip install pgeocode
```


# Quickstart

**Postal code queries**

```python
>>> import pgeocode

>>> nomi = pgeocode.Nominatim('fr')
>>> nomi.query_postal_code("75013")
postal_code               75013
country_code                 FR
place_name             Paris 13
state_name        Île-de-France
state_code                   11
county_name               Paris
county_code                  75
community_name            Paris
community_code              751
latitude                48.8322
longitude                2.3561
accuracy                      5

>>> nomi.query_postal_code(["75013", "69006"])
      postal_code place_name            state_name  latitude  longitude
0       75013   Paris 13         Île-de-France   48.8322     2.3561
1       69006    Lyon 06  Auvergne-Rhône-Alpes   45.7679     4.8506
```

**Place name queries**

```python
>>> import pgeocode

>>> nomi = pgeocode.Nominatim('fr')
>>> nomi.query_location("Antibes", top_k=3)
    country_code  postal_code place_name                  state_name  state_code  ... community_name community_code latitude longitude  accuracy
49553           FR        06160    Antibes  Provence-Alpes-Côte d'Azur        93.0  ...         Grasse            061  43.5858    7.1083         5
49787           FR        06600    Antibes  Provence-Alpes-Côte d'Azur        93.0  ...         Grasse            061  43.5858    7.1083         5
49788           FR  06601 CEDEX    Antibes  Provence-Alpes-Côte d'Azur        93.0  ...         Grasse            061  43.5858    7.1083         5

>>> nomi.query_location("Straassborg", top_k=3, fuzzy_threshold=80)
    country_code  postal_code  place_name state_name  state_code  ... community_name community_code latitude longitude  accuracy
25461           FR        67000  Strasbourg  Grand Est        44.0  ...     Strasbourg            678  48.5839    7.7455         5
25462           FR  67001 CEDEX  Strasbourg  Grand Est        44.0  ...     Strasbourg            678  48.5839    7.7455         5
25463           FR  67002 CEDEX  Strasbourg  Grand Est        44.0  ...     Strasbourg            678  48.5839    7.7455         5
```

**Distance calculations**

```python
>>> dist = pgeocode.GeoDistance('fr')
>>> dist.query_postal_code("75013", "69006")
389.156
>>> dist.query_postal_code(["75013", "75014", "75015"], ["69006", "69005", "69004"])
array([ 389.15648697,  390.12577967,  390.49857655])
```

# Geocoding format

The result of a geo-localistion query is a `pandas.DataFrame` with the
following columns,

-   `country_code`: iso country code, 2 characters
-   `postal_code` : postal code
-   `place_name` : place name (e.g. town, city etc)
-   `state_name` : 1. order subdivision (state)
-   `state_code` : 1. order subdivision (state)
-   `county_name` : 2. order subdivision (county/province)
-   `county_code` : 2. order subdivision (county/province)
-   `community_name` : 3. order subdivision (community)
-   `community_code` : 3. order subdivision (community)
-   `latitude` : estimated latitude (wgs84)
-   `longitude` : estimated longitude (wgs84)
-   `accuracy` : accuracy of lat/lng from 1=estimated to 6=centroid

# Configuration and defaults

**Storage directory**

Defaults to `~/.cache/pgeocode`, it is the directory where data is
downloaded for later consumption. It can be changed using the
environment variable `PGEOCODE_DATA_DIR`, i.e.
`export PGEOCODE_DATA_DIR=/tmp/pgeocode_data`.

**Data sources**

Data sources are provided as a list in the `pgeocode.DOWNLOAD_URL`
variable. The default value is,

``` python
DOWNLOAD_URL = [
    "https://download.geonames.org/export/zip/{country}.zip",
    "https://symerio.github.io/postal-codes-data/data/geonames/{country}.txt",
]
```

Data sources are tried from first to last until one works. Here the
second link is a mirror of the first.

It is also possible to extend this variable with third party data
sources, as long as they follow the same format. See for instance
[postal-codes-data](https://github.com/symerio/postal-codes-data/tree/master/data/geonames)
repository for examples of data files.

# License

The pgeocode package is distributed under the 3-clause BSD license.

The pgeocode package is maintained by [Symerio](https://www.symerio.com).

# Supported countries

The list of countries available in the GeoNames database, with the
corresponding country codes, are given below,

Andorra (AD), Argentina (AR), American Samoa (AS), Austria (AT),
Australia (AU), Åland Islands (AX), Azerbaijan (AZ), Bangladesh (BD),
Belgium (BE), Bulgaria (BG), Bermuda (BM), Brazil (BR), Belarus (BY),
Canada (CA), Switzerland (CH), Chile (CL), Colombia (CO), Costa Rica
(CR), Cyprus (CY), Czechia (CZ), Germany (DE), Denmark (DK), Dominican
Republic (DO), Algeria (DZ), Estonia (EE), Spain (ES), Finland (FI),
Federated States of Micronesia (FM), Faroe Islands (FO), France (FR),
United Kingdom of Great Britain and Northern Ireland (GB), French Guiana
(GF), Guernsey (GG), Greenland (GL), Guadeloupe (GP), Guatemala (GT),
Guam (GU), Croatia (HR), Haiti (HT), Hungary (HU), Ireland (IE), Isle of
Man (IM), India (IN), Iceland (IS), Italy (IT), Jersey (JE), Japan (JP),
Republic of Korea (KR), Liechtenstein (LI), Sri Lanka (LK), Lithuania
(LT), Luxembourg (LU), Latvia (LV), Monaco (MC), Republic of Moldova
(MD), Marshall Islands (MH), The former Yugoslav Republic of Macedonia
(MK), Northern Mariana Islands (MP), Martinique (MQ), Malta (MT), Malawi
(MW), Mexico (MX), Malaysia (MY), New Caledonia (NC), Netherlands (NL),
Norway (NO), New Zealand (NZ), Peru (PE), Philippines (PH), Pakistan
(PK), Poland (PL), Saint Pierre and Miquelon (PM), Puerto Rico (PR),
Portugal (PT), Palau (PW), Réunion (RE), Romania (RO), Serbia (RS),
Russian Federation (RU), Sweden (SE), Singapore (SG), Slovenia (SI),
Svalbard and Jan Mayen Islands (SJ), Slovakia (SK), San Marino (SM),
Thailand (TH), Turkey (TR), Ukraine (UA), United States of America (US),
Uruguay (UY), Holy See (VA), United States Virgin Islands (VI), Wallis
and Futuna Islands (WF), Mayotte (YT), South Africa (ZA)

See [GeoNames database](http://download.geonames.org/export/zip/) for
more information.
