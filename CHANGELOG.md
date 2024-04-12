# Release notes
## Version 0.5.0
*April 12, 2024*

 - Change to minimum supported Python version to 3.10. Run unit test with pandas 2.0.
   [#74](https://github.com/symerio/pgeocode/pull/74)

## Version 0.4.1

*September 7, 2022*

 - NA county_code is valid, not Nan
   [#74](https://github.com/symerio/pgeocode/pull/74)

## Version 0.4.0

*December 13, 2022*


 - The minimum supported Python version is updated to Python 3.8
   [#65](https://github.com/symerio/pgeocode/pull/65)
 - Fix error in latitude grouping when creating a unique postcode index.
   With this fix `Nominatim(.., unique=True)` correctly computes the average
   latitude for each postcode (if multiple localities share the same postcode),
   instead of taking the first latitude value.
   [#62](https://github.com/symerio/pgeocode/pull/62)

 - The default folder to store downloaded data is changed to `~/.cache/pgeocode/`.
   This default can still be changed by setting the `PGEOCODE_DATA_DIR` environment variable.
   [#51](https://github.com/symerio/pgeocode/pull/51)

 - Implemented `Nominatim.query_location` to query place names with text search
   Fuzzy search is supported if an optional extra dependency `thefuzz` is installed.
   [#59](https://github.com/symerio/pgeocode/pull/59)
 - Add more countries that are now supported by GeoNames including: AZ (Azerbaijan), CL (Chile), CY (Cyprus), EE (Estonia), FM (Federated States of Micronesia), HT (Haiti), KR (Republic of Korea), MW (Malawi), PE (Peru), PW (Palau), RS (Serbia), SG (Singapore)
   [#66](https://github.com/symerio/pgeocode/pull/66)


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
