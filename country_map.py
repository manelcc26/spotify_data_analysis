def alpha2_to_alpha3(alpha2_code):
      alpha2_to_alpha3_map = {
      "AF": "AFG",  # Afghanistan
      "AX": "ALA",  # Åland Islands
      "AL": "ALB",  # Albania
      "DZ": "DZA",  # Algeria
      "AS": "ASM",  # American Samoa
      "AD": "AND",  # Andorra
      "AO": "AGO",  # Angola
      "AI": "AIA",  # Anguilla
      "AQ": "ATA",  # Antarctica
      "AG": "ATG",  # Antigua and Barbuda
      "AR": "ARG",  # Argentina
      "AM": "ARM",  # Armenia
      "AW": "ABW",  # Aruba
      "AU": "AUS",  # Australia
      "AT": "AUT",  # Austria
      "AZ": "AZE",  # Azerbaijan
      "BS": "BHS",  # Bahamas
      "BH": "BHR",  # Bahrain
      "BD": "BGD",  # Bangladesh
      "BB": "BRB",  # Barbados
      "BY": "BLR",  # Belarus
      "BE": "BEL",  # Belgium
      "BZ": "BLZ",  # Belize
      "BJ": "BEN",  # Benin
      "BM": "BMU",  # Bermuda
      "BT": "BTN",  # Bhutan
      "BO": "BOL",  # Bolivia
      "BQ": "BES",  # Bonaire, Sint Eustatius, and Saba
      "BA": "BIH",  # Bosnia and Herzegovina
      "BW": "BWA",  # Botswana
      "BV": "BVT",  # Bouvet Island
      "BR": "BRA",  # Brazil
      "IO": "IOT",  # British Indian Ocean Territory
      "BN": "BRN",  # Brunei Darussalam
      "BG": "BGR",  # Bulgaria
      "BF": "BFA",  # Burkina Faso
      "BI": "BDI",  # Burundi
      "CV": "CPV",  # Cabo Verde
      "KH": "KHM",  # Cambodia
      "CM": "CMR",  # Cameroon
      "CA": "CAN",  # Canada
      "KY": "CYM",  # Cayman Islands
      "CF": "CAF",  # Central African Republic
      "TD": "TCD",  # Chad
      "CL": "CHL",  # Chile
      "CN": "CHN",  # China
      "CX": "CXR",  # Christmas Island
      "CC": "CCK",  # Cocos (Keeling) Islands
      "CO": "COL",  # Colombia
      "KM": "COM",  # Comoros
      "CG": "COG",  # Congo (Brazzaville)
      "CD": "COD",  # Congo (Kinshasa)
      "CK": "COK",  # Cook Islands
      "CR": "CRI",  # Costa Rica
      "CI": "CIV",  # Côte d'Ivoire
      "HR": "HRV",  # Croatia
      "CU": "CUB",  # Cuba
      "CW": "CUW",  # Curaçao
      "CY": "CYP",  # Cyprus
      "CZ": "CZE",  # Czech Republic
      "DK": "DNK",  # Denmark
      "DJ": "DJI",  # Djibouti
      "DM": "DMA",  # Dominica
      "DO": "DOM",  # Dominican Republic
      "EC": "ECU",  # Ecuador
      "EG": "EGY",  # Egypt
      "SV": "SLV",  # El Salvador
      "GQ": "GNQ",  # Equatorial Guinea
      "ER": "ERI",  # Eritrea
      "EE": "EST",  # Estonia
      "SZ": "SWZ",  # Eswatini
      "ET": "ETH",  # Ethiopia
      "FK": "FLK",  # Falkland Islands
      "FO": "FRO",  # Faroe Islands
      "FJ": "FJI",  # Fiji
      "FI": "FIN",  # Finland
      "FR": "FRA",  # France
      "GF": "GUF",  # French Guiana
      "PF": "PYF",  # French Polynesia
      "TF": "ATF",  # French Southern Territories
      "GA": "GAB",  # Gabon
      "GM": "GMB",  # Gambia
      "GE": "GEO",  # Georgia
      "DE": "DEU",  # Germany
      "GH": "GHA",  # Ghana
      "GI": "GIB",  # Gibraltar
      "GR": "GRC",  # Greece
      "GL": "GRL",  # Greenland
      "GD": "GRD",  # Grenada
      "GP": "GLP",  # Guadeloupe
      "GU": "GUM",  # Guam
      "GT": "GTM",  # Guatemala
      "GG": "GGY",  # Guernsey
      "GN": "GIN",  # Guinea
      "GW": "GNB",  # Guinea-Bissau
      "GY": "GUY",  # Guyana
      "HT": "HTI",  # Haiti
      "HM": "HMD",  # Heard Island and McDonald Islands
      "VA": "VAT",  # Holy See
      "HN": "HND",  # Honduras
      "HK": "HKG",  # Hong Kong
      "HU": "HUN",  # Hungary
      "IS": "ISL",  # Iceland
      "IN": "IND",  # India
      "ID": "IDN",  # Indonesia
      "IR": "IRN",  # Iran
      "IQ": "IRQ",  # Iraq
      "IE": "IRL",  # Ireland
      "IM": "IMN",  # Isle of Man
      "IL": "ISR",  # Israel
      "IT": "ITA",  # Italy
      "JM": "JAM",  # Jamaica
      "JP": "JPN",  # Japan
      "JE": "JEY",  # Jersey
      "JO": "JOR",  # Jordan
      "KZ": "KAZ",  # Kazakhstan
      "KE": "KEN",  # Kenya
      "KI": "KIR",  # Kiribati
      "KP": "PRK",  # North Korea
      "KR": "KOR",  # South Korea
      "KW": "KWT",  # Kuwait
      "KG": "KGZ",  # Kyrgyzstan
      "LA": "LAO",  # Laos
      "LV": "LVA",  # Latvia
      "LB": "LBN",  # Lebanon
      "LS": "LSO",  # Lesotho
      "LR": "LBR",  # Liberia
      "LY": "LBY",  # Libya
      "LI": "LIE",  # Liechtenstein
      "LT": "LTU",  # Lithuania
      "LU": "LUX",  # Luxembourg
      "MO": "MAC",  # Macao
      "MG": "MDG",  # Madagascar
      "MW": "MWI",  # Malawi
      "MY": "MYS",  # Malaysia
      "MV": "MDV",  # Maldives
      "ML": "MLI",  # Mali
      "MT": "MLT",  # Malta
      "MH": "MHL",  # Marshall Islands
      "MQ": "MTQ",  # Martinique
      "MR": "MRT",  # Mauritania
      "MU": "MUS",  # Mauritius
      "YT": "MYT",  # Mayotte
      "MX": "MEX",  # Mexico
      "FM": "FSM",  # Micronesia
      "MD": "MDA",  # Moldova
      "MC": "MCO",  # Monaco
      "MN": "MNG",  # Mongolia
      "ME": "MNE",  # Montenegro
      "MS": "MSR",  # Montserrat
      "MA": "MAR",  # Morocco
      "MZ": "MOZ",  # Mozambique
      "MM": "MMR",  # Myanmar
      "NA": "NAM",  # Namibia
      "NR": "NRU",  # Nauru
      "NP": "NPL",  # Nepal
      "NL": "NLD",  # Netherlands
      "NC": "NCL",  # New Caledonia
      "NZ": "NZL",  # New Zealand
      "NI": "NIC",  # Nicaragua
      "NE": "NER",  # Niger
      "NG": "NGA",  # Nigeria
      "NU": "NIU",  # Niue
      "NF": "NFK",  # Norfolk Island
      "MK": "MKD",  # North Macedonia
      "MP": "MNP",  # Northern Mariana Islands
      "NO": "NOR",  # Norway
      "OM": "OMN",  # Oman
      "PK": "PAK",  # Pakistan
      "PW": "PLW",  # Palau
      "PS": "PSE",  # Palestine
      "PA": "PAN",  # Panama
      "PG": "PNG",  # Papua New Guinea
      "PY": "PRY",  # Paraguay
      "PE": "PER",  # Peru
      "PH": "PHL",  # Philippines
      "PN": "PCN",  # Pitcairn
      "PL": "POL",  # Poland
      "PT": "PRT",  # Portugal
      "PR": "PRI",  # Puerto Rico
      "QA": "QAT",  # Qatar
      "RE": "REU",  # Réunion
      "RO": "ROU",  # Romania
      "RU": "RUS",  # Russia
      "RW": "RWA",  # Rwanda
      "BL": "BLM",  # Saint Barthélemy
      "SH": "SHN",  # Saint Helena, Ascension, and Tristan da Cunha
      "KN": "KNA",  # Saint Kitts and Nevis
      "LC": "LCA",  # Saint Lucia
      "MF": "MAF",  # Saint Martin (French part)
      "PM": "SPM",  # Saint Pierre and Miquelon
      "VC": "VCT",  # Saint Vincent and the Grenadines
      "WS": "WSM",  # Samoa
      "SM": "SMR",  # San Marino
      "ST": "STP",  # Sao Tome and Principe
      "SA": "SAU",  # Saudi Arabia
      "SN": "SEN",  # Senegal
      "RS": "SRB",  # Serbia
      "SC": "SYC",  # Seychelles
      "SL": "SLE",  # Sierra Leone
      "SG": "SGP",  # Singapore
      "SX": "SXM",  # Sint Maarten (Dutch part)
      "SK": "SVK",  # Slovakia
      "SI": "SVN",  # Slovenia
      "SB": "SLB",  # Solomon Islands
      "SO": "SOM",  # Somalia
      "ZA": "ZAF",  # South Africa
      "GS": "SGS",  # South Georgia and the South Sandwich Islands
      "SS": "SSD",  # South Sudan
      "ES": "ESP",  # Spain
      "LK": "LKA",  # Sri Lanka
      "SD": "SDN",  # Sudan
      "SR": "SUR",  # Suriname
      "SJ": "SJM",  # Svalbard and Jan Mayen
      "SE": "SWE",  # Sweden
      "CH": "CHE",  # Switzerland
      "SY": "SYR",  # Syria
      "TW": "TWN",  # Taiwan
      "TJ": "TJK",  # Tajikistan
      "TZ": "TZA",  # Tanzania
      "TH": "THA",  # Thailand
      "TL": "TLS",  # Timor-Leste
      "TG": "TGO",  # Togo
      "TK": "TKL",  # Tokelau
      "TO": "TON",  # Tonga
      "TT": "TTO",  # Trinidad and Tobago
      "TN": "TUN",  # Tunisia
      "TR": "TUR",  # Turkey
      "TM": "TKM",  # Turkmenistan
      "TC": "TCA",  # Turks and Caicos Islands
      "TV": "TUV",  # Tuvalu
      "UG": "UGA",  # Uganda
      "UA": "UKR",  # Ukraine
      "AE": "ARE",  # United Arab Emirates
      "GB": "GBR",  # United Kingdom
      "US": "USA",  # United States
      "UM": "UMI",  # United States Minor Outlying Islands
      "UY": "URY",  # Uruguay
      "UZ": "UZB",  # Uzbekistan
      "VU": "VUT",  # Vanuatu
      "VE": "VEN",  # Venezuela
      "VN": "VNM",  # Vietnam
      "VG": "VGB",  # British Virgin Islands
      "VI": "VIR",  # U.S. Virgin Islands
      "WF": "WLF",  # Wallis and Futuna
      "EH": "ESH",  # Western Sahara
      "YE": "YEM",  # Yemen
      "ZM": "ZMB",  # Zambia
      "ZW": "ZWE",  # Zimbabwe
      }
      return alpha2_to_alpha3_map.get(alpha2_code, alpha2_code)


def alpha2_to_name(alpha2_code):
    alpha2_to_name_map = {
        "AF": "Afghanistan",
        "AX": "Åland Islands",
        "AL": "Albania",
        "DZ": "Algeria",
        "AS": "American Samoa",
        "AD": "Andorra",
        "AO": "Angola",
        "AI": "Anguilla",
        "AQ": "Antarctica",
        "AG": "Antigua and Barbuda",
        "AR": "Argentina",
        "AM": "Armenia",
        "AW": "Aruba",
        "AU": "Australia",
        "AT": "Austria",
        "AZ": "Azerbaijan",
        "BS": "Bahamas",
        "BH": "Bahrain",
        "BD": "Bangladesh",
        "BB": "Barbados",
        "BY": "Belarus",
        "BE": "Belgium",
        "BZ": "Belize",
        "BJ": "Benin",
        "BM": "Bermuda",
        "BT": "Bhutan",
        "BO": "Bolivia",
        "BQ": "Bonaire, Sint Eustatius and Saba",
        "BA": "Bosnia and Herzegovina",
        "BW": "Botswana",
        "BV": "Bouvet Island",
        "BR": "Brazil",
        "IO": "British Indian Ocean Territory",
        "BN": "Brunei Darussalam",
        "BG": "Bulgaria",
        "BF": "Burkina Faso",
        "BI": "Burundi",
        "CV": "Cabo Verde",
        "KH": "Cambodia",
        "CM": "Cameroon",
        "CA": "Canada",
        "KY": "Cayman Islands",
        "CF": "Central African Republic",
        "TD": "Chad",
        "CL": "Chile",
        "CN": "China",
        "CX": "Christmas Island",
        "CC": "Cocos (Keeling) Islands",
        "CO": "Colombia",
        "KM": "Comoros",
        "CG": "Congo (Brazzaville)",
        "CD": "Congo (Kinshasa)",
        "CK": "Cook Islands",
        "CR": "Costa Rica",
        "CI": "Côte d'Ivoire",
        "HR": "Croatia",
        "CU": "Cuba",
        "CW": "Curaçao",
        "CY": "Cyprus",
        "CZ": "Czech Republic",
        "DK": "Denmark",
        "DJ": "Djibouti",
        "DM": "Dominica",
        "DO": "Dominican Republic",
        "EC": "Ecuador",
        "EG": "Egypt",
        "SV": "El Salvador",
        "GQ": "Equatorial Guinea",
        "ER": "Eritrea",
        "EE": "Estonia",
        "SZ": "Eswatini",
        "ET": "Ethiopia",
        "FK": "Falkland Islands",
        "FO": "Faroe Islands",
        "FJ": "Fiji",
        "FI": "Finland",
        "FR": "France",
        "GF": "French Guiana",
        "PF": "French Polynesia",
        "TF": "French Southern Territories",
        "GA": "Gabon",
        "GM": "Gambia",
        "GE": "Georgia",
        "DE": "Germany",
        "GH": "Ghana",
        "GI": "Gibraltar",
        "GR": "Greece",
        "GL": "Greenland",
        "GD": "Grenada",
        "GP": "Guadeloupe",
        "GU": "Guam",
        "GT": "Guatemala",
        "GG": "Guernsey",
        "GN": "Guinea",
        "GW": "Guinea-Bissau",
        "GY": "Guyana",
        "HT": "Haiti",
        "HM": "Heard Island and McDonald Islands",
        "VA": "Holy See",
        "HN": "Honduras",
        "HK": "Hong Kong",
        "HU": "Hungary",
        "IS": "Iceland",
        "IN": "India",
        "ID": "Indonesia",
        "IR": "Iran",
        "IQ": "Iraq",
        "IE": "Ireland",
        "IM": "Isle of Man",
        "IL": "Israel",
        "IT": "Italy",
        "JM": "Jamaica",
        "JP": "Japan",
        "JE": "Jersey",
        "JO": "Jordan",
        "KZ": "Kazakhstan",
        "KE": "Kenya",
        "KI": "Kiribati",
        "KP": "North Korea",
        "KR": "South Korea",
        "KW": "Kuwait",
        "KG": "Kyrgyzstan",
        "LA": "Laos",
        "LV": "Latvia",
        "LB": "Lebanon",
        "LS": "Lesotho",
        "LR": "Liberia",
        "LY": "Libya",
        "LI": "Liechtenstein",
        "LT": "Lithuania",
        "LU": "Luxembourg",
        "MO": "Macao",
        "MG": "Madagascar",
        "MW": "Malawi",
        "MY": "Malaysia",
        "MV": "Maldives",
        "ML": "Mali",
        "MT": "Malta",
        "MH": "Marshall Islands",
        "MQ": "Martinique",
        "MR": "Mauritania",
        "MU": "Mauritius",
        "YT": "Mayotte",
        "MX": "Mexico",
        "FM": "Micronesia",
        "MD": "Moldova",
        "MC": "Monaco",
        "MN": "Mongolia",
        "ME": "Montenegro",
        "MS": "Montserrat",
        "MA": "Morocco",
        "MZ": "Mozambique",
        "MM": "Myanmar",
        "NA": "Namibia",
        "NR": "Nauru",
        "NP": "Nepal",
        "NL": "Netherlands",
        "NC": "New Caledonia",
        "NZ": "New Zealand",
        "NI": "Nicaragua",
        "NE": "Niger",
        "NG": "Nigeria",
        "NU": "Niue",
        "NF": "Norfolk Island",
        "MK": "North Macedonia",
        "MP": "Northern Mariana Islands",
        "NO": "Norway",
        "OM": "Oman",
        "PK": "Pakistan",
        "PW": "Palau",
        "PS": "Palestine",
        "PA": "Panama",
        "PG": "Papua New Guinea",
        "PY": "Paraguay",
        "PE": "Peru",
        "PH": "Philippines",
        "PN": "Pitcairn",
        "PL": "Poland",
        "PT": "Portugal",
        "PR": "Puerto Rico",
        "QA": "Qatar",
        "RE": "Réunion",
        "RO": "Romania",
        "RU": "Russia",
        "RW": "Rwanda",
        "BL": "Saint Barthélemy",
        "SH": "Saint Helena, Ascension and Tristan da Cunha",
        "KN": "Saint Kitts and Nevis",
        "LC": "Saint Lucia",
        "MF": "Saint Martin (French part)",
        "PM": "Saint Pierre and Miquelon",
        "VC": "Saint Vincent and the Grenadines",
        "WS": "Samoa",
        "SM": "San Marino",
        "ST": "Sao Tome and Principe",
        "SA": "Saudi Arabia",
        "SN": "Senegal",
        "RS": "Serbia",
        "SC": "Seychelles",
        "SL": "Sierra Leone",
        "SG": "Singapore",
        "SX": "Sint Maarten (Dutch part)",
        "SK": "Slovakia",
        "SI": "Slovenia",
        "SB": "Solomon Islands",
        "SO": "Somalia",
        "ZA": "South Africa",
        "GS": "South Georgia and the South Sandwich Islands",
        "SS": "South Sudan",
        "ES": "Spain",
        "LK": "Sri Lanka",
        "SD": "Sudan",
        "SR": "Suriname",
        "SJ": "Svalbard and Jan Mayen",
        "SE": "Sweden",
        "CH": "Switzerland",
        "SY": "Syria",
        "TW": "Taiwan",
        "TJ": "Tajikistan",
        "TZ": "Tanzania",
        "TH": "Thailand",
        "TL": "Timor-Leste",
        "TG": "Togo",
        "TK": "Tokelau",
        "TO": "Tonga",
        "TT": "Trinidad and Tobago",
        "TN": "Tunisia",
        "TR": "Turkey",
        "TM": "Turkmenistan",
        "TC": "Turks and Caicos Islands",
        "TV": "Tuvalu",
        "UG": "Uganda",
        "UA": "Ukraine",
        "AE": "United Arab Emirates",
        "GB": "United Kingdom",
        "US": "United States",
        "UM": "United States Minor Outlying Islands",
        "UY": "Uruguay",
        "UZ": "Uzbekistan",
        "VU": "Vanuatu",
        "VE": "Venezuela",
        "VN": "Vietnam",
        "VG": "British Virgin Islands",
        "VI": "U.S. Virgin Islands",
        "WF": "Wallis and Futuna",
        "EH": "Western Sahara",
        "YE": "Yemen",
        "ZM": "Zambia",
        "ZW": "Zimbabwe",
    }
    return alpha2_to_name_map.get(alpha2_code, alpha2_code)