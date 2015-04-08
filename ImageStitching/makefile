CXXFLAGS=-std=c++11 -O3 -Wall -fopenmp -ffast-math
EXECUTABLES += ImageStitching 
LIBS := -lexiv2 -lboost_system -lboost_filesystem -lopencv_core -lopencv_calib3d -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_stitching
SUBDIRS := \
src \

RM := rm -rf

# All of the sources participating in the build
-include src/subdir.mk

# All Target
all: $(EXECUTABLES)

# Tool invocations
ImageStitching: $(OBJS) 
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ $(CXXFLAGS) -o "ImageStitching" $(OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(EXECUTABLES)$(OBJS)$(CPP_DEPS) 
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

