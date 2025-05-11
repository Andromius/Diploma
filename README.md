# SacsA

## Architektura

- Pipes & Filters
- Flask, pytorch (torchivision)

## Spuštění
- *(sudo)* docker-compose -f docker-compose.dev.yml up (-d pro background)

## Smazání
- *(sudo)* docker-compose -f docker-compose.dev.yml down --rmi all

## Testy
- coverage run -m pytest
- coverage report --ignore-errors --omit="tests/*" 


## *poznamky pod carou*
- .env moc nefunguje takze ty env variably si musis exportnout sam
- pripadne docker

## Závislosti
- pip install -r requirements.txt