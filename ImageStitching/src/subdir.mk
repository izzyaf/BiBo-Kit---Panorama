CXXFLAGS=-std=c++11 -O3 -Wall -fopenmp -ffast-math

# Inputs and outputs 
CPP_SRCS += \
./src/Stitcher.cpp \
./src/main.cpp 

O_SRCS += \
./src/Stitcher.o \
./src/main.o 

OBJS += \
./src/Stitcher.o \
./src/main.o 

CPP_DEPS += \
./src/Stitcher.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ $(CXXFLAGS) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


