#include <iostream>
#include "OptoDAQ.h"
#include "OptoDAQDescriptor.h"
#include "OptoPacket6D.h"
#include "OptoDAQWatcher.h"




int main()
{
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
		OptoConfig config = OptoConfig(100, 4, 0);
		success = daqs[i].SendConfig(config); // Set up the speed to 100 Hz and filtering to 15 Hz
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
	for (std::size_t i = 0; i < count; ++i) {
		std::cout << "10 packets from DAQ " << i + 1 <<":" << std::endl;
		if (daqs[i].IsValid()) {
			daqs[i].GetPackets6D(&packets, true); // blocking call, waits for 10 packets
		}
		// Show the captured packets Fx value in newtons
		std::size_t size = packets.GetSize(); // It should be 10.
		for (std::size_t j = 0; j < size; ++j) {
			OptoPacket6D p = packets.GetPacket(j);
			if (p.IsValid()) {
				std::cout << "Fx: " << p.GetFxInNewton() << std::endl;
			}
		}
		packets.Clear(); // Empty the container for the next DAQ
	}


	// Clean-up

	delete[] daqs;

	// Wait for user input
	char ch;
	std::cin >> ch;

    return 0;
}
