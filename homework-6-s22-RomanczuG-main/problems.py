import re

def problem1(searchstring):

    p = re.compile(r'^((\+1\s|\+52\s)\(?\d{3}((\)\s)|-)?)?\d{3}-?\d{4}$')

    match = p.search(searchstring)
    if(match):
        return True
    else:
        return False


    """
    Match phone numbers.

    :param searchstring: string
    :return: True or False
    """
    pass
        
def problem2(searchstring):

    p = re.compile(r'(\d+(\s[A-Z][a-z]*)+)\s(Ave\.|St\.|Rd\.|Dr\.)')
    match = p.search(searchstring)

    return match.group(1)

    """
    Extract street name from address.

    :param searchstring: string
    :return: string
    """
    pass
    
def problem3(searchstring):

    p = re.compile(r'\d+((\s[A-Z][a-z]*)+\s)(Ave\.|St\.|Rd\.|Dr\.)')
    match = p.search(searchstring)
    
    street = match.group(1)
    print(street)
    result = re.sub(street + "(?=(Ave\.|St\.|Rd\.|Dr\.))", street[::-1], searchstring)
    print(result)
    return result
    """
    Garble Street name.

    :param searchstring: string
    :return: string
    """
    pass


if __name__ == '__main__' :
    print("\nProblem 1:")
    print("Answer correct?", problem1('+1 765-494-4600') == True)
    print("Answer correct?", problem1('+52 765-494-4600 ') == False)
    print("Answer correct?", problem1('+1 (765) 494 4600') == False)
    print("Answer correct?", problem1('+52 (765) 494-4600') == True)
    print("Answer correct?", problem1('+52 7654944600') == True)
    print("Answer correct?", problem1('494-4600') == True)

    print("\nProblem 2:")
    print("Answer correct?",problem2('Please flip your wallet at 465 Northwestern Ave.') == "465 Northwestern")
    print("Answer correct?",problem2('Meet me at 201 South First St. at noon') == "201 South First")
    print("Answer correct?",problem2('Type "404 Not Found St" on your phone at 201 South First St. at noon') == "201 South First")
    print("Answer correct?",problem2("123 Mayb3 Y0u 222 Did not th1nk 333 This Through Rd. Did Y0u Ave.") == "333 This Through")
    print("\nProblem 3:")
    print("Answer correct?",problem3('The EE building is at 465 Northwestern Ave.') == "The EE building is at 465 nretsewhtroN Ave.")
    print("Answer correct?",problem3('Meet me at 201 South First St. at noon') == "Meet me at 201 tsriF htuoS St. at noon")






