import { Component } from '../types';

export const mockComponents: Component[] = [
  {
    id: '1',
    partNumber: 'ATmega328P',
    manufacturer: 'Microchip',
    category: 'Microcontrollers',
    description: '8-bit AVR Microcontroller with 32K Bytes In-System Programmable Flash',
    datasheetUrl: '/assets/sample.pdf',
    specifications: {
      'Operating Voltage': '1.8-5.5V',
      'CPU Speed': 'Up to 20MHz',
      'Flash Memory': '32KB',
      'SRAM': '2KB'
    }
  },
  {
    id: '2',
    partNumber: 'LM317T',
    manufacturer: 'Texas Instruments',
    category: 'Voltage Regulators',
    description: '1.5A Adjustable Voltage Regulator',
    datasheetUrl: '/assets/sample.pdf',
    specifications: {
      'Input Voltage': '3-40V',
      'Output Voltage': '1.25-37V',
      'Maximum Current': '1.5A',
      'Package': 'TO-220'
    }
  },
  {
    id: '3',
    partNumber: 'ESP32-WROOM-32',
    manufacturer: 'Espressif',
    category: 'WiFi Modules',
    description: 'WiFi & Bluetooth Combo Module',
    datasheetUrl: '/assets/sample.pdf',
    specifications: {
      'WiFi': '2.4 GHz',
      'Bluetooth': 'v4.2',
      'Flash Memory': '4MB',
      'RAM': '520KB'
    }
  }
];