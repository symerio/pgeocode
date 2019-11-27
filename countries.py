import warnings
from typing import Set


class Countries:
    def __init__(self):
        self.__countries_valid = {"AD", "AR", "AS", "AT", "AU", "AX", "BD", "BE", "BG", "BM",
                                  "BR", "BY", "CA", "CH", "CO", "CR", "CZ", "DE", "DK", "DO",
                                  "DZ", "ES", "FI", "FO", "FR", "GB", "GB_full", "GF", "GG",
                                  "GL", "GP", "GT", "GU", "HR", "HU", "IE", "IM", "IN", "IS",
                                  "IT", "JE", "JP", "LI", "LK", "LT", "LU", "LV", "MC", "MD",
                                  "MH", "MK", "MP", "MQ", "MT", "MX", "MY", "NC", "NL", "NO",
                                  "NZ", "PH", "PK", "PL", "PM", "PR", "PT", "RE", "RO", "RU",
                                  "SE", "SI", "SJ", "SK", "SM", "TH", "TR", "UA", "US", "UY",
                                  "VA", "VI", "WF", "YT", "ZA"}

    @property
    def countries_valid(self) -> Set[str]:
        return self.__countries_valid

    def get_clean_country(self, country: str) -> str:
        country = country.upper()
        if country == 'AR':
            warnings.warn('The Argentina data file contains the first 5 positions of the postal code.')
        if country == 'GB_FULL':
            return 'GB_full'
        if country in self.__countries_valid:
            return country
        else:
            raise ValueError(('country={} is not a known country code. '
                              'See the README for a list of supported '
                              'countries')
                             .format(country))

    def get_clean_country_for_download_path(self, country: str) -> str:
        country = self.get_clean_country(country)
        if country == 'GB_full':
            return 'GB_full.csv'
        else:
            return country
