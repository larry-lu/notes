# Regular expression in Python

## Part 1

More information can be found at [Python documentation (3.6)](https://docs.python.org/3.6/library/re.html). This note is created based on the tutorial on [Geekforgeeks](https://www.geeksforgeeks.org/regular-expression-python-examples-set-1/).

Regular Expressions (RE) specifies a set of strings pattern that matches it. And the library `re` is useful to use regular expression in Python.

There are 14 meta-characters for regular expression:

```
\   Used to escape the special meaning of character
[]  A character class
^   The beginning
$   The end
.   Any character except newline
?   Zero or one occurrence.
|   OR (characters separated by it)
*   Any number of occurrences (including 0)
+   One or more occurrences
{}  number of occurrences of a preceding RE to match
()  A group of REs
```

#### Function compile()

RE are compiled into pattern objects, which have methods for operations such as searching and string substitutions. 

```python
import re

#compile() creates RE character class [a-e] 
#which is equivalent to [abcde]
p = re.compile('[a-e]')

#findall() method searches for RE and returns 
#a list of found items.
print (p.findall('The quick brown fox jumps over the lazy dog.'))
```

Output:

```python
['e', 'a', 'd', 'b', 'e', 'a']
```

Understanding the Output:

First occurrence is ‘e’ in “Aye” and not ‘A’, as it being Case Sensitive.
Next Occurrence is ‘a’ in “said”, then ‘d’ in “said”, followed by ‘b’ and ‘e’ in “Gibenson”, the Last ‘a’ matches with “Stark”.

Metacharacter blackslash ‘\’ has a very important role as it signals various sequences. If the blackslash is to be used without its special meaning as metacharcter, use ’\\\’ 

```
\d  Matches any decimal digit, this is equivalent to the set class [0-9].
\D  Matches any non-digit character.
\s  Matches any whitespace character.
\S  Matches any non-whitespace character
\w  Matches any alphanumeric character, this is 
    equivalent to the class [a-zA-Z0-9_].
\W  Matches any non-alphanumeric character. 
\s  any white space character, ',', or '.'.
```

Example:

```python
import re

p = re.compile('\d')

## \d is equivalent to [0-9]. 
print (p.findall('I went to him at 11 A.M. on 4th July 1886'))

# \d+ will match a group on [0-9], group of one or greater size 
p = re.compile('\d+')
print (p.findall('I went to him at 11 A.M. on 4th July 1886'))
```

Output:

```python
['1', '1', '4', '1', '8', '8', '6']
['11', '4', '1886']
```

```python
import re

# \w is equivalent to [a-zA-Z0-9_]. 
p = re.compile('\w') 
print(p.findall("He said * in some_lang.")) 

# \w+ matches to group of alphanumeric charcter. 
p = re.compile('\w+') 
print(p.findall("I went to him at 11 A.M., he said *** in some_language.")) 
  
# \W matches to non alphanumeric characters. 
p = re.compile('\W') 
print(p.findall("he said *** in some_language.")) 
```

Output:

```python
['H', 'e', 's', 'a', 'i', 'd', 'i', 'n', 's', 'o', 'm', 'e', '_', 'l', 'a', 'n', 'g']
['I', 'went', 'to', 'him', 'at', '11', 'A', 'M', 'he', 'said', 'in', 'some_language']
[' ', ' ', '*', '*', '*', ' ', ' ', '.']
```

```python
import re

# '*' replaces the no. of occurrence of a character.
p = re.compile('ab*')
print (p.findall('ababbaabbb'))
```

Output:

```python
['ab', 'abb', 'a', 'abbb']
```

Understanding the Output:
Our RE is ab*, which ‘a’ accompanied by any no. of ‘b’s, starting from 0.
Output ‘ab’, is valid because of singe ‘a’ accompanied by single ‘b’.
Output ‘abb’, is valid because of singe ‘a’ accompanied by 2 ‘b’.
Output ‘a’, is valid because of singe ‘a’ accompanied by 0 ‘b’.
Output ‘abbb’, is valid because of singe ‘a’ accompanied by 3 ‘b’.

#### Function split()

Split string by the occurrences of a character or a pattern. Upon finding the pattern, the remaining characters from the string are returned as a resulting list.

Syntax:

```python
re.split(pattern, string, maxsplit=0, flags = 0)
```

The First parameter, pattern denotes the regular expression, string is the given string in which pattern will be searched for and in which splitting occurs, maxsplit if not provided is considered to be zero ‘0’, and if any nonzero value is provided, then at most that many splits occurs. If maxsplit = 1, then the string will split once only, resulting in a list of length 2. The flags are very useful and can help to shorten code, they are not necessary parameters, eg: flags = re.IGNORECASE, In this split, case will be ignored.

Example:

```python
from re import split

# '\W+' denotes Non-Alphanumeric Characters or group of characters 
# Upon finding ',' or whitespace ' ', the split()
#splits the string from that point 
print(split('\W+', 'Words, words , Words')) 
print(split('\W+', "Word's words Words")) 
  
# Here ':', ' ' ,',' are not AlphaNumeric thus, the point where splitting occurs 
print(split('\W+', 'On 12th Jan 2016, at 11:02 AM')) 
  
# '\d+' denotes Numeric Characters or group of characters 
# Spliting occurs at '12', '2016', '11', '02' only 
print(split('\d+', 'On 12th Jan 2016, at 11:02 AM')) 
```

Output:

```python
['Words', 'words', 'Words']
['Word', 's', 'words', 'Words']
['On', '12th', 'Jan', '2016', 'at', '11', '02', 'AM']
['On ', 'th Jan ', ', at ', ':', ' AM']
```

```python
import re 
  
# Splitting will occurs only once, at '12', returned list will have length 2 
print(re.split('\d+', 'On 12th Jan 2016, at 11:02 AM', 1)) 
  
# 'Boy' and 'boy' will be treated same when flags = re.IGNORECASE 
print(re.split('[a-f]+', 'Aey, Boy oh boy, come here', flags = re.IGNORECASE)) 
print(re.split('[a-f]+', 'Aey, Boy oh boy, come here')) 

```

Output:

```python
['On ', 'th Jan 2016, at 11:02 AM']
['', 'y, ', 'oy oh ', 'oy, ', 'om', ' h', 'r', '']
['A', 'y, Boy oh ', 'oy, ', 'om', ' h', 'r', '']
```
#### Function sub()

Syntax:

```python
re.sub(pattern, repl, string, count=0, flags=0)
```

The ‘sub’ in the function stands for SubString, a certain regular expression pattern is searched in the given string(3rd parameter), and upon finding the substring pattern is replaced by by repl(2nd parameter), count checks and maintains the number of times this occurs. 

```python

import re 
  
# Regular Expression pattern 'ub' matches the string at "Subject" and "Uber". 
# As the CASE has been ignored, using Flag, 'ub' should match twice with the string 
# Upon matching, 'ub' is replaced by '~*' in "Subject", and in "Uber", 'Ub' is replaced. 
print(re.sub('ub', '~*' , 'Subject has Uber booked already', flags = re.IGNORECASE)) 
  
# Consider the Case Senstivity, 'Ub' in "Uber", will not be reaplced. 
print(re.sub('ub', '~*' , 'Subject has Uber booked already')) 
  
# As count has been given value 1, the maximum times replacement occurs is 1 
print(re.sub('ub', '~*' , 'Subject has Uber booked already', count=1, flags = re.IGNORECASE)) 
  
# 'r' before the patter denotes RE, \s is for start and end of a String. 
print(re.sub(r'\sAND\s', ' & ', 'Baked Beans And Spam', flags=re.IGNORECASE)) 
```

#### Function subn()

Syntax:

```python
re.subn(pattern, repl, string, count=0, flags=0)
```

subn() is similar to sub() in all ways, except in its way to providing output. It returns a tuple with count of total of replacement and the new string rather than just the string.

```python
import re 
print(re.subn('ub', '~*' , 'Subject has Uber booked already')) 
t = re.subn('ub', '~*' , 'Subject has Uber booked already', flags = re.IGNORECASE) 
print(t) 
print(len(t)) 
  
# This will give same output as sub() would have  
print(t[0]) 
```

Output:

```
('S~*ject has Uber booked already', 1)
('S~*ject has ~*er booked already', 2)
2
S~*ject has ~*er booked already
```

#### Function escape()

Syntax:

```python
re.escape(string)
```

Return string with all non-alphanumerics backslashed, this is useful if you want to match an arbitrary literal string that may have regular expression metacharacters in it.

```python

import re 
  
# escape() returns a string with BackSlash '\', before every Non-Alphanumeric Character 
# In 1st case only ' ', is not alphanumeric 
# In 2nd case, ' ', caret '^', '-', '[]', '\' are not alphanumeric 
print(re.escape("This is Awseome even 1 AM")) 
print(re.escape("I Asked what is this [a-9], he said \t ^WoW"))
```

Output:

```
This\ is\ Awesome\ even\ 1\ AM
I\ Asked\ what\ is\ this\ \[a\-9\]\,\ he\ said\ \	\ \^WoW
```

## Part 2

More information can be found at [Python documentation (3.6)](https://docs.python.org/3.6/library/re.html). This note is created based on the tutorial on [Geekforgeeks](https://www.geeksforgeeks.org/regular-expressions-python-set-1-search-match-find/).

The module re provides support for regular expressions in Python. Below are main methods in this module.


#### Function search()

This method either returns None (if the pattern doesn’t match), or a re.MatchObject that contains information about the matching part of the string. This method stops after the first match, so this is best suited for testing a regular expression more than extracting data.

```python
import re 
  
# Lets use a regular expression to match a date string 
# in the form of Month name followed by day number 
regex = r"([a-zA-Z]+) (\d+)"
  
match = re.search(regex, "I was born on June 24") 
  
if match != None: 
  
    # We reach here when the expression "([a-zA-Z]+) (\d+)" 
    # matches the date string. 
  
    # This will print [14, 21), since it matches at index 14 
    # and ends at 21.  
    print ("Match at index {0}, {1}".format(match.start(), match.end())) 
  
    # We use group() method to get all the matches and 
    # captured groups. The groups contain the matched values. 
    # In particular: 
    #    match.group(0) always returns the fully matched string 
    #    match.group(1) match.group(2), ... return the capture 
    #    groups in order from left to right in the input string 
    #    match.group() is equivalent to match.group(0) 
  
    # So this will print "June 24" 
    print ("Full match: {0}".format(match.group(0)) )
  
    # So this will print "June" 
    print ("Month: {0}".format(match.group(1)) )
  
    # So this will print "24" 
    print ("Day: {0}".format(match.group(2))) 
  
else: 
    print ("The regex pattern does not match.")
```

Output:

```
Match at index 14, 21
Full match: June 24
Month: June
Day: 24
```

#### Function match()

This function attempts to match pattern to whole string. The re.match function returns a match object on success, None on failure. 

Syntax:

```python
re.match(pattern, string, flags=0)
```
where pattern is the RE to be matched, string is the String where pattern is searched, and flags is different flags which can be modified using bitwise OR (|). 

```python
# A Python program to demonstrate working 
# of re.match(). 
import re 
  
# a sample function that uses regular expressions 
# to find month and day of a date. 
def findMonthAndDate(string): 
      
    regex = r"([a-zA-Z]+) (\d+)"
    match = re.match(regex, string) 
      
    if match == None:  
        print "Not a valid date"
        return
  
    print "Given Data: %s" % (match.group()) 
    print "Month: %s" % (match.group(1)) 
    print "Day: %s" % (match.group(2)) 
  
      
# Driver Code 
findMonthAndDate("Jun 24") 
print("") 
findMonthAndDate("I was born on June 24") 
```

#### Function findall()

Return all non-overlapping matches of pattern in string, as a list of strings. The string is scanned left-to-right, and matches are returned in the order found.

```python

# A Python program to demonstrate working of 
# findall() 
import re 
  
# A sample text string where regular expression  
# is searched. 
string  = """Hello my Number is 123456789 and 
             my friend's number is 987654321"""
  
# A sample regular expression to find digits. 
regex = '\d+'             
  
match = re.findall(regex, string) 
print(match) 
  
# This example is contributed by Ayush Saluja. 
```

Output:

```python
['123456789', '987654321']
```

#### example: extract email addresses

```python



# extract all email addresses and add them into the resulting set
new_emails = set(re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text, re.I))

```