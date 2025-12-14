#!/usr/bin/env python3
import os
os.chdir('c:\\Users\\Win10\\Documents\\GitHub\\Analiza-i-Obrada')
from scraper_production import NFLScraper
s = NFLScraper(verbose=False)
qbs = ['AlleJo02', 'JackLa00', 'BurrJo01', 'HurtJa00', 'HerbJu00', 'RodgAa00', 
       'MahoPa00', 'PresDa01', 'GoffJa00', 'PurdBr00', 'TagoTu00', 'StroCJ00', 
       'DarnSa00', 'LawrTr00']
for qb in qbs:
    s.scrape(qb)
