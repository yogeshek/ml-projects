# class regularuser:
#     def __init__(self, name: str):
#         if not isinstance(name, str):
#             raise TypeError(f"name must be a string but got: {type(name)}")
#         self.name = name
# name_input = input("please give a name: ")
# try:
#     user = regularuser(
#         name = name_input
#     )
#     print(f"✅ User created with name: {user.name}")
# except TypeError as e:
#     print(f"\n❌ TypeError: {e}")
    
    
# class Dog:
#     def test(raw, name):      # SPECIAL METHOD
#         """Runs ONCE automatically when object created"""
#         raw.name = name
#         print(f"Dog {name} was born!")
    
#     def bark(raw1):                # REGULAR METHOD
#         """Runs only when explicitly called"""
#         print(f"{raw1.name} says Woof!")
    
#     def __str__(self):             # SPECIAL METHOD
#         """Runs when printing"""
#         return f"Dog named {self.name}"

# # __init__ runs automatically here ↓
# my_dog = Dog("Fido")  # Prints: Dog Fido was born!

# # Regular method must be called explicitly ↓
# my_dog.bark()  # Prints: Fido says Woof!

# # __str__ runs automatically when printing ↓
# print(my_dog)  # Prints: Dog named Fido

class student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    def __str__(self):
        # Called when you use print() or str()
        return f"{self.name} (Grade: {self.grade})"
alice =student('Alice','A')
print(alice)