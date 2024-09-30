
# Dataset description

We collected data for all the devices connected to the network. Depending on the experiment performed, the traffic was captured for the entire network (Active and Idle), a part of the network (Scenarios) or a single device, filtered by a distinctive ID such as a MAC address (Power, Interactions and Attacks).


However, the experiment used in our case study was the Interactions experiment.


## Interactions experiment




In this experiment, all of the devices in our environment
were powered on individually, and the network traffic was
captured in isolation for a duration of two
minutes after which the device is plugged into the power source. 
The device’s MAC address is set in Wireshark as the capture filter, 
not to include data produced by the router or other non-IoT devices 
that may have been connected to the network.

The capture continues for an additional 15 seconds to 
capture any delayed packets that remain from the activity. 
Possible interactions via the companion app were discovered 
by either looking through the documentation provided by the 
manufacturer or experimenting with the app.

Four types of Interactions were considered:
  - Physical (Using buttons or other manual commands on device or using native voice commands), 
  - LAN App (Using companion app on a mobile device, while being on same network as the IoT device), 
  - WAN App (Using companion app on a mobile device, while being on different network from the IoT Device), and 
  - Voice (Using a voice assistant, like Google assistant or Amazon Alexa, to command the IoT Device).


Each experiment was repeated n = 3 times in order to generate sufficient
network packets. 

### Naming conversion
Each packet naming was based on a combincation of the four types of interaction experiments conducted and the device name. 
```bash
  devicenameLanPhysical_ExperimentNumber.csv
  devicenameWanPhysical_ExperimentNumber.csv
```
This is different based on the device type.
- Camera is a combincation of LAN, and WAN with each of the physical activities.
    - Physical
        - Photo
        - Recording
        - Watch
    In total a camera will have a 18 different packets if each experiment is conducted 3 times.
    ```bash
    amcrestcamLANPHOTO_1.csv
    amcrestcamWANRECORDING_2.csv
    amcrestcamLANWATCH_3.csv
    ```
- Audio is a combincation of LAN, WAN, and voice with each of the physical activities.
    - Physical
        - Volume off
        - Volume on
    In total an audio device will have a 24 different packets if each experiment is conducted 3 times.
    ```bash
    echostudioLANVOLUMEOFF_1.csv
    echostudioLOCALVOLUMEON_2.csv
    echostudioVOICEVOLUMEON_3.csv
    echostudioWANVOLUMEOFF_1.csv
    ```
- Home automation is a combincation of LAN, WAN, local, and voice with each of the physical activities.
    - Physical
        - On
        - Off
    In total a home automation device will have a 24 different packets if each experiment is conducted 3 times.
    ```bash
    amazonplugALEXAON_1.csv
    amazonplugLANOFF_2.csv
    amazonplugWANOFF_3.csv
    amazonplugLOCALON_1.csv
    ```
    
    