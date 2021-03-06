gestureNames = {100: 'Rest',
				101: 'Index Flexion',
				102: 'Index Extension',
				103: 'Middle Flexion',
				104: 'Middle Extension',
				105: 'Ring Flexion',
				106: 'Ring Extension',
				107: 'Pinky Flexion',
				108: 'Pinky Extension',
				109: 'Thumb Flexion',
				110: 'Thumb Extension',
				111: 'Thumb Adduction',
				112: 'Thumb Abduction',
				201: 'One',
				202: 'Two',
				203: 'Three',
				204: 'Four',
				205: 'Five',
				206: 'Thumb Up',
				207: 'Fist',
				208: 'Flat'}

gestureGroupNames = {0: 'All Gestures',
				1: 'Single-DOF',
				2: 'Single-DOF Flexion',
				3: 'Single-DOF Extension',
				4: 'Single-DOF Thumb',
				5: 'Multi-DOF',
				6: 'Multi-DOF Count',
				7: 'Multi-DOF Other',
				8: 'One',
				9: 'Two',
				10: 'Fist',
				11: 'Index Flexion',
				12: 'Index Extension',
				13: 'Middle Flexion',
				14: 'Middle Extension',
				15: 'Thumb Flexion',
				16: 'Thumb Extension'}

gestureGroupMembers = {0: sorted(gestureNames.keys()),
				1: range(100,113),
				2: [101, 103, 105, 107],
				3: [102, 104, 106, 108],
				4: [100, 109, 110, 111, 112],
				5: range(201,209),
				6: range(201,206),
				7: range(206,209),
				8: [201],
				9: [202],
				10: [207],
				11: [101],
				12: [102],
				13: [103],
				14: [104],
				15: [109],
				16: [110]}