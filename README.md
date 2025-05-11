# SacsA

## Architektura

- Pipes & Filters
- Flask, pytorch (torchivision)

## Spuštění
- *(sudo)* docker-compose -f docker-compose.dev.yml up (-d pro background)

## Smazání
- *(sudo)* docker-compose -f docker-compose.dev.yml down --rmi all

## Testy
![CI](https://github.com/Andromius/Diploma/actions/workflows/ci.yml/badge.svg)

- ./run_tests.sh


## *poznamky pod carou*
- .env moc nefunguje takze ty env variably si musis exportnout sam
- pripadne docker

## Závislosti
- pip install -r requirements.txt