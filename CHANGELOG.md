# Release notes

## Version 0.3.0

*October 23, 2020*

 - Add support for a fallback mechanism for data sources
 - Set the default fallback URL to https://symerio.github.io/postal-codes-data/
   (only used when the main GeoNames server has availability issues).
 - Add support for data sources in .txt format (non zipped)
 - Document how to use custom data sources

## Version 0.2.1

*February 21, 2019*

 - Fix ``Nominatim`` for AS, LT, VA countries.
 - Drop Python 2.7 support.
 - Allow custom download locations.

## Version 0.2.0

*December 24, 2019*

 - Fix download URL.
 - Drop Python 2.7 support.

## Version 0.1.2

*November 8, 2019*

 - Allow looking up all the locations for a postcode with the
   `unique=False` parameter of `Nominatim`.
 - Fix handling of Candian postal codes

## Version 0.1.1

*November 8, 2018*

 - Fix compatibility with pandas >=0.23

## Version 0.1.0

*August 28, 2018*

Initial release
