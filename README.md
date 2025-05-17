# SacsA

## Architektura

- Pipes & Filters
- Flask, pytorch (torchivision)

## Spuštění
- *(sudo)* docker compose -f docker-compose.dev.yml up (-d pro background)

## Smazání
- *(sudo)* docker compose -f docker-compose.dev.yml down --rmi all

## Testy
![CI](https://github.com/Andromius/Diploma/actions/workflows/python-app.yml/badge.svg)

- *(sudo)* docker compose -f docker-compose.test.yml up
- *(sudo)* docker compose -f docker-compose.test.yml down --rmi all


## *poznamky pod carou*
- .env moc nefunguje takze ty env variably si musis exportnout sam
- pripadne docker

## Závislosti
- pip install -r requirements.txt
