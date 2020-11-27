# fernet.py
# Utils to encrypt/decrypt sensitive keys
# TODO
# This is necesary till we find a way to add keys into trains.conf
# file automatically

from cryptography.fernet import Fernet


FERNET_KEY = "sPW7NoxeCWt4rVHZNb5Acfln-dN3kgoGAPV04hVap-s="


def encrypt(message: bytes, key: bytes) -> bytes:
    return Fernet(key).encrypt(message)


def decrypt(token: bytes, key: bytes) -> bytes:
    return Fernet(key).decrypt(token)
