from synergos.functions.hello import say_hello

def test_helloworld_no_params():
    assert say_hello() == "Hello, World!"
    #assert True

def test_helloworld_with_param():
    assert say_hello('Everyone') == 'Hello, Everyone!'
    #assert True