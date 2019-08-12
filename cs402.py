from numpy import mat
from numpy import transpose
from numpy import linalg
from math import log
import codecs
#from sympy import Matrix #undelete is the sympy module is installed
import matplotlib.pyplot as plt
import numpy as np
from random import randint


##############################################
##############################################
#### Some handy alphabets
##############################################
Alphabet68="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;: \n"
Alphabet64="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789. "
Alphabet26="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
Alphabet27="ABCDEFGHIJKLMNOPQRSTUVWXYZ "

##############################################
##############################################
#### Conversion from string to list of integers
##############################################
def StringToIntList(Alphabet,m):
    M=[]
    space=Alphabet.find(' ')
    for x in m:
        y=Alphabet.find(x)
        if y>=0:
             M.append(y)
        else:
             M.append(space)
    return M
##############################################
##############################################
#### Conversion from list of integers to string
##############################################
def IntListToString(Alphabet,M):
    m=[]
    for x in M: m.append(Alphabet[x])
    return ''.join(m)


##############################################
##############################################
#### A useful class for a generic cipher system
##############################################
class Cipher:

    ##########################################
    def __init__(self):
        def stringToIntList(file):
           f=open(file,"r")
           m=f.read()
           f.close()
           return StringToIntList(self.Alphabet,m)
        def intListToString(M):
           return IntListToString(self.Alphabet,M)
        def encipher(file):
           M=self.stringToIntList(file)
           M=self.EncipherIntList(M)
           text=self.intListToString(M)
           return text
        def decipher(file):
           M=self.stringToIntList(file)
           M=self.DecipherIntList(M)
           text=self.intListToString(M)
           return text
        self.stringToIntList=stringToIntList
        self.intListToString=intListToString
        self.encipher=encipher
        self.decipher=decipher
    ##########################################

    ##To set/change the alphabet use:
    def setAlphabet(self,text):
        self.Alphabet=text

    ##To install the enciphering function use:
    def setEncipherIntList(self,function):
       self.EncipherIntList=function

    ##To install the deciphering function use:
    def setDecipherIntList(self,function):
       self.DecipherIntList=function

    ##To set/change the enciphering key use:
    def setEncipherKey(self,text):
       self.EncipherKey=text

    ##To set/change the deciphering key use:
    def setDecipherKey(self,text):
       self.DecipherKey=text


##############################################
##############################################
#### The affine cipher x --> ax + b 
#### Special cases include: Caeser cipher, Vignere Cipher, Hill Cipher
##############################################
AffineCipher=Cipher()
A=3
B=0
AffineCipher.setEncipherKey([A,B])
AffineCipher.setDecipherKey([A,B])
AffineCipher.setAlphabet(Alphabet68)
def fn(M):
   a=AffineCipher.EncipherKey[0]
   b=AffineCipher.EncipherKey[1]
   if type(a) is int: a=mat([[a]])
   if type(b) is int: b=mat([[b]])
   N=len(AffineCipher.Alphabet)
   D=len(a)
   spc=AffineCipher.Alphabet.find(' ')
   if spc==-1: spc=0
   while not (len(M) % D) ==0 : M.append(spc)
   m=[]
   for i in range(0,int(len(M)/D)) :
       v=mat(M[D*i:D*i+D])
       v=(v*transpose(a)+transpose(b)) % N
       v=v.tolist()
       m=m+v[0]    
   return m
AffineCipher.setEncipherIntList(fn)
def fn(M):
   a=AffineCipher.DecipherKey[0]
   b=AffineCipher.DecipherKey[1]
   if type(a) is int: a=mat([[a]])
   if type(b) is int: b=mat([[b]])
   N=len(AffineCipher.Alphabet)
   D=len(a)
   spc=AffineCipher.Alphabet.find(' ')
   if spc==-1: spc=0
   while not (len(M) % D) ==0 : M.append(spc)
   m=[]
   for i in range(0,int(-D+len(M)/D)) :
       v=mat(M[D*i:D*i+D])
       v=(v*transpose(a)+transpose(b)) % N
       v=v.tolist()
       m=m+v[0]
   return m
AffineCipher.setDecipherIntList(fn)

##############################################
##############################################
#### A function to display the frequenty of 
#### symbols in a string
##############################################
def FrequencyHistogram(x):
    alph=set(x)
    alph=''.join(alph)
    alph=sorted(alph)
    data=[]
    for c in alph: data.append(x.count(c))
    print data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(alph))
    width=0.35
    rects1 = ax.bar(ind, data, width, color='lightblue')
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(alph)

    plt.show()


##############################################
##############################################
#### A function for detecting if text is in English
#### Modified from http://inventwithpython.com/hacking (BSD Licensed)
#### This function could certainly be improved!
##############################################
UPPERLETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LETTERS_AND_SPACE = UPPERLETTERS + UPPERLETTERS.lower() + ' \t\n'

def loadDictionary():
#    dictionaryFile = open('dictionary.txt',"r")
    dictionaryFile = codecs.open('dictionary.txt',"r",encoding="utf-8",errors='ignore')
    englishWords = {}
    for word in dictionaryFile.read().split('\n'):
        englishWords[word] = None
    dictionaryFile.close()
    return englishWords

ENGLISH_WORDS = loadDictionary()

def getEnglishCount(message):
    message = message.lower()
    message = removeNonLetters(message)
    possibleWords = message.split()

    if possibleWords == []:
        return 0.0 # no words at all, so return 0.0

    matches = 0
    for word in possibleWords:
        if word in ENGLISH_WORDS:
            matches += 1
    return float(matches) / len(possibleWords)

def removeNonLetters(message):
    lettersOnly = []
    for symbol in message:
        if symbol in LETTERS_AND_SPACE:
            lettersOnly.append(symbol)
    return ''.join(lettersOnly)

def isEnglish(message, wordPercentage=75, letterPercentage=85):
    # By default, 75% of the words must exist in the dictionary file, and
    # 85% of all the characters in the message must be letters or spaces
    # (not punctuation or numbers).
    wordsMatch = getEnglishCount(message) * 100 >= wordPercentage
    numLetters = len(removeNonLetters(message))
    messageLettersPercentage = float(numLetters) / len(message) * 100
    lettersMatch = messageLettersPercentage >= letterPercentage
    return wordsMatch and lettersMatch

##############################################
##############################################
#### Returns a list consisting of every d-th entry 
#### in the list L, starting at position s in L.
##############################################
def lst(L,s,d):
       M=[]
       for i in range(0,int(len(L)/d)-s):
          M.append(L[s+d*i])
       return M

##############################################
##############################################
#### Returns the number of times the most frequent 
#### letter occurs in the string x.
##############################################
def MostFrequentLetter(x):
       L=[]
       for a in set(x):
         L.append(x.count(a))
       return max(L)

##############################################
##############################################
#### Returns the inverse of an integer mod N. 
##############################################
def egcd(aa, b):
    a= (aa % b)
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)
def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

##############################################
##############################################
#### Returns the inverse of a square matrix of 
#### integers modulo N.
##############################################
def InverseMatModN(AA,N):
       A=Matrix(AA)
       #return ( mat( modinv(int(linalg.det(A)),N)*A.adjugate() ) %N  )
       return ( mat( modinv(int(A.det()),N)*A.adjugate() ) %N  )

##############################################
##############################################
#### Convert a list of decimal integers to a
#### binary list. Each integer is converted to
#### a binary list of length n.
##############################################
def dec2bin(L,n):
    B=[]
    for x in L:
              b=bin(x)
              b=b[2:]
              for k in range(0,n-len(b)): B.append(0)
              for k in b:
                        if k=='1': B.append(1)
                        else: B.append(0)
    return B

##############################################
##############################################
#### Convert a binary list to a list of decimal
#### integers. We break the binary list into binary
#### blocks each of length n, and then convert each
#### to an integer.
##############################################
def bin2dec(B,n):
    if not (len(B) % n == 0): print("Fail"); return False
    L=[]
    for k in range(0,int(len(B)/n)):
       w=0
       for i in range(0,n):
           w=w+B[n*k+i]*(2**(n-1-i))
       L.append(w)
    return L

##############################################
##############################################
#### Convert a binary list to a single integer, 
#### namely the decimal integer corresponding to
#### the binary number represented by the list.
##############################################
def binList2int(B):
    return int("".join(str(x) for x in B),2)

##############################################
##############################################
#### Convert a single integer to a binary list 
##############################################
def int2binList(n):
    a=bin(n)
    return [int(x) for x in a[2:len(a)]]



##############################################
##############################################
#### A binary linear feedback shift register with
#### connection polynomial
####       1 + c_1X + c_2 X^2 + ... + c_nX^n
#### where c_n is non-zero. The function inputs a
#### list K of the indices i of the non-zero
#### coefficients c_i and a binary list V of length
#### n. It returns a binary list W of length n.
##############################################
def LFSR(K,V):
        w=0
        ln=len(K)
        lnv=len(V)
        for i in range(0,ln):
           w=(w+V[lnv-K[i]]) % 2
        W=V[1:]
        W.append(w)
        return W

def PeriodLFSR(K,S):
        T=LFSR(K,S)
        cnt=1
        while not T==S:
          T=LFSR(K,T)
          cnt=cnt+1
        return cnt


##############################################
##############################################
#### The RSA cipher
##############################################
RSA=Cipher()
p=304250263527209
q=23768741896345550770650537601358309
e=3  #Need gcd(e,(p-1)(q-1))==1
N=p*q
d=4821097323782215625692549251331855329314609896043
RSA.setEncipherKey([N,e])
RSA.setDecipherKey([N,d])
RSA.setAlphabet(Alphabet64)
def fn(M):
    N=RSA.EncipherKey[0]
    e=RSA.EncipherKey[1]
    ln=int(log(len(RSA.Alphabet),2))
    mm=dec2bin(M,ln)
    mm=binList2int(mm)
    mm= pow(mm,e,N) 
    mm=int2binList(mm)
    while not 0 == (len(mm) % ln) : mm= [0]+mm
    mm=bin2dec(mm,ln)
    return mm 
RSA.setEncipherIntList(fn)
def fn(M):
    N=RSA.DecipherKey[0]
    e=RSA.DecipherKey[1]
    ln=int(log(len(RSA.Alphabet),2))
    mm=dec2bin(M,ln)
    mm=binList2int(mm)
    mm= pow(mm,e,N) 
    mm=int2binList(mm)
    while not 0 == (len(mm) % ln) : mm= [0]+mm
    mm=bin2dec(mm,ln)
    return mm
RSA.setDecipherIntList(fn)

##############################################
##############################################
#### A Naive trial division method for finding the
#### factors of an integer. I took the verbatim from
#### StackOverflow
##############################################
def factors(n):  
    return set(reduce(list.__add__,([i, n//i] for i in xrange(1, int(pow(n, 0.5) + 1)) if n % i == 0)))

##############################################
##############################################
#### Fermat Primality Test: a composite non-Carmichael
#### number returns true with probability <(1/2)^k
##############################################
def IsProbablyPrime(n,k):
    for i in range(0,k):
        a=randint(1,n)
        if not pow(a,n-1,n)==1: return False
    return True

##############################################
##############################################
#### Fermat Primality Test is used to identify
#### what is probably the next prime number.
##############################################
def ProbablyNextPrime(n):
    m=n
    while not IsProbablyPrime(m,100): m=m+1
    return m
