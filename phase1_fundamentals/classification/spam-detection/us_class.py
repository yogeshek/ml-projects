class regularuser:
    def __init__(self, name: int):
        if not isinstance(name, int):
            raise TypeError(f"name must be a string but got: {type(name)}")
        self.name = name
name_input = input("please give a name: ")
try:
    user = regularuser(
        name = name_input
    )
    print(f"✅ User created with name: {user.name}")
except TypeError as e:
    print(f"\n❌ TypeError: {e}")