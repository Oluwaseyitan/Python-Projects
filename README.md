#Program is designed to calculate the area of a shape

print ("Hello There, the calculator is now available and running")

option = raw_input ("""What shape would you like to calculate?
Enter C for Circle or T for Triangle: """)

if option == "C":
  radius = float(raw_input("Enter radius: "))
  area = 3.14159 * radius**2
  print ("The area for the circle with radius %s is area %s") %(radius, area) 
elif option == 'T':
  base = float(raw_input ("enter base of triangle: "))
  height = float (raw_input ("Enter height of triangle: "))
  area = 0.5 * base * height
  print ("The area of the triangle with base %s and height %s is area %s ") % (base, height, area)
else:
  print ("Invalid Shape Entered")
  
print ("Thank you for using the Area calculator. It is now stopping.")
  


PLANNING A TRIP:

def hotel_cost(nights):
  hotel_cost = 140
  return 140*nights 
 
def plane_ride_cost(city):
  if city == "Charlotte":
    return 183
  elif city == "Tampa":
    return 220
  elif city == "Pittsburgh":
    return 222
  elif city == "Los Angeles":
    return 475


