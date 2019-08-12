from cs402 import AffineCipher
from cs402 import isEnglish

C = AffineCipher
C.setAlphabet("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMONPQRSTUVWXYZ0123456789.,;: \n")


def CaesarDecipher(C):
  for i in range(48,58):
    C.setDecipherKey([1,i])
    txt = C.decipher("cipher.txt")
    if isEnglish(txt):
      print( "\n")
      print((str(i)))
CaesarDecipher(C)

def CaesarEncipher(C):
  C.setEncipherKey([1,-56])
  enc = C.encipher("plain.txt")
  enc_file = open("enc_q1.txt", "w+")
  enc_file.write(enc)
  print(enc)
CaesarEncipher(C)

def CaesarReDecipher(C):
  C.setDecipherKey([1,56])
  print(C.decipher("enc_q1.txt"))

CaesarReDecipher(C)
