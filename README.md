# Python-Projects

#Program is designed to calculate the area of a circle or a triangle

print ("Hello There, the calculator is now available and running")

option = raw_input ("""What shape would you like to calculate?
Enter C for Circle or T for Triangle: """)

if option == "C":
  radius = float(raw_input("Enter radius: "))
  area = 3.14159 * radius**2
  print ("The area for the Triangle with radius %s is area %s") %(radius, area) 
