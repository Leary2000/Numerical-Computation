def Vigenere(plaintext, key):
    ciphertext = []
    keylength = len(key)
    plaintext = plaintext.replace(" ", "")  # remove all spaces from the plaintext

    for i in range(len(plaintext)):
        shift = ord(key[i % keylength]) - ord('A') #ord gets ASCII value of char
        x = (ord(plaintext[i]) - ord('A') + shift) % 26 
        x += ord('A')
        ciphertext.append(chr(x))
    return "" . join(ciphertext)


def modInverse(n, modulus): 
    n=n%modulus
    for x in range(1, modulus): 
        if ((n*x)%modulus == 1): 
            return x 
    return None

def letterFrequency(String):
    frequency = {}
    for char in String:
        frequency[char] = frequency.get(char, 0) + 1
    return frequency

def index_of_coincidence(s: str) -> float:
    N = len(s)
    freqs = [s.count(c) for c in set(s)]
    return sum([f*(f-1) for f in freqs]) / (N*(N-1))



print(index_of_coincidence("U S Z H L M T C OAYHIZUSJESKGJEER"))
print(Vigenere("MAKETHEPASSMARKFIFTY", "PANDORA") )
print(modInverse(365,10007))




