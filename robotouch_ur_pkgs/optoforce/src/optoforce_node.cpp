#include <iostream>
#include "OptoDAQ.h"
#include "OptoDAQDescriptor.h"
#include "OptoPacket6D.h"
#include "OptoDAQWatcher.h"

// ROS Headers
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/WrenchStamped.h"

#include <sstream>

int main(int argc, char ** argv )
{
	ros::init(argc, argv, "force_torque_usb");
	ros::NodeHandle n;
	ros::Publisher chatter_pub = n.advertise<geometry_msgs::WrenchStamped>("wrench_onrobot", 10);
	ros::Rate loop_rate(500);

	/*
	Create an OptoDAQWatcher instance that can enumerate connected DAQs via USB
	*/
	OptoDAQWatcher watcher;
	watcher.Start();  // Start the watcher on a different thread


	OptoDAQDescriptor descriptors[16];

	/*
	Trying to get connected DAQs (max.: 16, it can be changed up to 64)
	*/
	std::size_t count = watcher.GetConnectedDAQs(descriptors, 16, true);
	while (count == 0) {
		count = watcher.GetConnectedDAQs(descriptors, 16, true);
	}
	// int count   = 1;

	/*
	Show information about connected DAQs
	*/
	for (std::size_t i = 0; i < count; ++i) {
		std::cout << "Information about Connected DAQ (" << i + 1 << "):" << std::endl;
		std::cout << "Connected on port: "<<descriptors[i].GetAddress()<<std::endl;
		std::cout << "Protocol version: " << descriptors[i].GetProtocolVersion() << std::endl;
		std::cout << "S/N:" << descriptors[i].GetSerialNumber() << std::endl;
		std::cout << "Type name:" << descriptors[i].GetTypeName() << std::endl;
		std::cout << "-----------------------" << std::endl;
	}


	// Open all the connected DAQs
	OptoDAQ  * daqs = new OptoDAQ[count];
	for (std::size_t i = 0; i < count; ++i) {
		daqs[i].SetOptoDAQDescriptor(descriptors[i]);
		bool success = daqs[i].Open();
		if (success == false) {
			std::cout << i + 1 << ". DAQ could not be opened!" << std::endl;
			continue;
		}
		OptoConfig config = OptoConfig(500, 4, 0);
		success = daqs[i].SendConfig(config); // Set up the speed to 500 Hz and filtering to 15 Hz

		//zero the hardware
		// TODO : add a service to dynamically zero the FT sensor
		config = OptoConfig(500, 4, 255);
		success = daqs[i].SendConfig(config);
		if (success) {
			std::cout << i + 1 << ". DAQ successfully configured." << std::endl;
		}
		else {
			std::cout << i + 1 << ". DAQ could not be configured." << std::endl;
			continue;
		}
		daqs[i].RequestSensitivityReport(); // This call is a must
	}

	// Create a container that can hold 10 6D packets
	OptoPackets6D packets(10);


	// Get 10 packets from every opened DAQs
	while (ros::ok())
	{
		geometry_msgs::WrenchStamped msg; 

		for (std::size_t i = 0; i < count; ++i) {
			//std::cout << "10 packets from DAQ " << i + 1 <<":" << std::endl;
			if (daqs[i].IsValid()) {
				daqs[i].GetPackets6D(&packets, true); // blocking call, waits for 10 packets
			}
			// Show the captured packets Fx value in newtons
			std::size_t size = packets.GetSize(); // It should be 10.
			for (std::size_t j = 0; j < size; ++j) {
				OptoPacket6D p = packets.GetPacket(j);
				if (p.IsValid()) {
					//std::cout << "Fx: " << p.GetFxInNewton() << std::endl;
					msg.header.stamp = ros::Time::now();
					msg.header.frame_id = "on_flange";
					msg.wrench.force.x = p.GetFxInNewton();
					msg.wrench.force.y = p.GetFyInNewton();
					msg.wrench.force.z = p.GetFzInNewton();
					msg.wrench.torque.x = p.GetTxInNewtonMeter();
					msg.wrench.torque.y = p.GetTyInNewtonMeter();
					msg.wrench.torque.z = p.GetTzInNewtonMeter();
					chatter_pub.publish(msg);
				}
			}
			packets.Clear(); // Empty the container for the next DAQ
			ros::spinOnce();
    		loop_rate.sleep();
		}
	}


	// Clean-up

	delete[] daqs;

	// Wait for user input
	char ch;
	std::cin >> ch;

    return 0;
}
