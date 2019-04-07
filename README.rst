pgeocode
========

|pypi| |rdfd| |travisci|

.. |pypi| image:: https://img.shields.io/pypi/v/pgeocode.svg
   :target: https://pypi.org/project/pgeocode/
   
.. |rdfd| image:: https://readthedocs.org/projects/pgeocode/badge/?version=latest
    :target: http://pgeocode.readthedocs.io/

.. |travisci| image:: https://travis-ci.org/symerio/pgeocode.svg?branch=master
   :target: https://travis-ci.org/symerio/pgeocode

Postal code geocoding and distance calculations

pgeocode is a Python library for high performance off-line querying of GPS coordinates, region name and municipality name
from postal codes. Distances between postal codes as well as general
distance queries are also supported.
The used `GeoNames <http://download.geonames.org/export/zip/>`_ database includes postal codes for 83 countries.

Currently, only queries within the same country are supported.

For additional documentation see `pgeocode.readthedocs.io <https://pgeocode.readthedocs.io>`_.


Installation
------------

pgeocode requires Python 2.7 or 3.5+ as well as ``numpy`` and ``pandas`` packages. It can be installed with,

.. code::

    pip install pgeocode

Quickstart
----------

**Postal code queries**

.. code:: python

    >>> import pgeocode

    >>> nomi = pgeocode.Nominatim('fr')
    >>> nomi.query_postal_code("75013")
    postal_code               75013
    country code                 FR
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

**Distance calculations**

.. code:: python

    >>> dist = pgeocode.GeoDistance('fr')
    >>> dist.query_postal_code("75013", "69006")
    389.156
    >>> dist.query_postal_code(["75013", "75014", "75015"], ["69006", "69005", "69004"])
    array([ 389.15648697,  390.12577967,  390.49857655])



Geocoding format
----------------

The result of a geo-localistion query is a ``pandas.DataFrame`` with the following columns,

* ``country code``: iso country code, 2 characters
* ``postal code`` : postal code
* ``place name``  : place name (e.g. town, city etc)
* ``state_name`` : 1. order subdivision (state)
* ``state_code`` : 1. order subdivision (state)
* ``county_name`` : 2. order subdivision (county/province)
* ``county_code`` : 2. order subdivision (county/province)
* ``community_name`` : 3. order subdivision (community)
* ``community_code`` : 3. order subdivision (community)
* ``latitude``    : estimated latitude (wgs84)
* ``longitude``   : estimated longitude (wgs84)
* ``accuracy``    : accuracy of lat/lng from 1=estimated to 6=centroid

License
-------

The pgeocode package is distributed under the 3-clause BSD license.


Supported countries
-------------------

The list of countries available in the GeoNames database, with the corresponding country codes, are given below,

Andorra (AD), Argentina (AR), American Samoa (AS), Austria (AT), Australia (AU), Åland Islands (AX), Bangladesh (BD), Belgium (BE), Bulgaria (BG), Bermuda (BM), Brazil (BR), Belarus (BY), Canada (CA), Switzerland (CH), Colombia (CO), Costa Rica (CR), Czechia (CZ), Germany (DE), Denmark (DK), Dominican Republic (DO), Algeria (DZ), Spain (ES), Finland (FI), Faroe Islands (FO), France (FR), United Kingdom of Great Britain and Northern Ireland (GB), French Guiana (GF), Guernsey (GG), Greenland (GL), Guadeloupe (GP), Guatemala (GT), Guam (GU), Croatia (HR), Hungary (HU), Ireland (IE), Isle of Man (IM), India (IN), Iceland (IS), Italy (IT), Jersey (JE), Japan (JP), Liechtenstein (LI), Sri Lanka (LK), Lithuania (LT), Luxembourg (LU), Latvia (LV), Monaco (MC), Republic of Moldova (MD), Marshall Islands (MH), The former Yugoslav Republic of Macedonia (MK), Northern Mariana Islands (MP), Martinique (MQ), Malta (MT), Mexico (MX), Malaysia (MY), New Caledonia (NC), Netherlands (NL), Norway (NO), New Zealand (NZ), Philippines (PH), Pakistan (PK), Poland (PL), Saint Pierre and Miquelon (PM), Puerto Rico (PR), Portugal (PT), Réunion (RE), Romania (RO), Russian Federation (RU), Sweden (SE), Slovenia (SI), Svalbard and Jan Mayen Islands (SJ), Slovakia (SK), San Marino (SM), Thailand (TH), Turkey (TR), Ukraine (UA), United States of America (US), Uruguay (UY), Holy See (VA), United States Virgin Islands (VI), Wallis and Futuna Islands (WF), Mayotte (YT), South Africa (ZA)

See `GeoNames database <http://download.geonames.org/export/zip/>`_ for more information.
