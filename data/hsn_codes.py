"""HSN/SAC code database with valid tax rates."""

# HSN code → (description, valid tax rates as percentages)
HSN_DATABASE: dict[str, tuple[str, list[float]]] = {
    # Food & Beverages
    "0201": ("Meat of bovine animals, fresh or chilled", [5.0]),
    "0401": ("Milk and cream, not concentrated", [0.0]),
    "0402": ("Milk and cream, concentrated or sweetened", [5.0]),
    "0713": ("Dried leguminous vegetables", [0.0]),
    "0902": ("Tea", [5.0]),
    "0901": ("Coffee", [5.0]),
    "1001": ("Wheat and meslin", [0.0]),
    "1006": ("Rice", [5.0]),
    "1101": ("Wheat or meslin flour", [5.0]),
    "1701": ("Cane or beet sugar", [5.0]),
    "1905": ("Bread, pastry, cakes, biscuits", [18.0]),
    "2106": ("Food preparations not elsewhere specified", [18.0]),
    "2201": ("Mineral waters and aerated waters", [18.0]),
    "2202": ("Sweetened or flavoured water", [28.0]),
    "2203": ("Beer made from malt", [28.0]),

    # Textiles
    "5208": ("Woven cotton fabrics", [5.0]),
    "6109": ("T-shirts and vests, knitted", [5.0, 12.0]),
    "6203": ("Men's suits, jackets, trousers", [12.0]),
    "6204": ("Women's suits, jackets, dresses", [12.0]),

    # Electronics
    "8471": ("Automatic data processing machines (computers)", [18.0]),
    "8517": ("Telephone sets including smartphones", [18.0]),
    "8528": ("Monitors and projectors, television receivers", [18.0]),
    "8443": ("Printing machinery, printers", [18.0]),
    "8415": ("Air conditioning machines", [28.0]),
    "8418": ("Refrigerators and freezers", [18.0]),
    "8450": ("Washing machines", [18.0]),
    "8516": ("Electric water heaters, hair dryers, irons", [18.0]),

    # Automobile
    "8703": ("Motor cars for transport of persons", [28.0]),
    "8711": ("Motorcycles and cycles with motors", [28.0]),
    "8714": ("Parts and accessories of motorcycles", [18.0, 28.0]),
    "4011": ("New pneumatic rubber tyres", [28.0]),

    # Pharma & Healthcare
    "3004": ("Medicaments in measured doses", [5.0, 12.0]),
    "3003": ("Medicaments not in measured doses", [12.0]),
    "3005": ("Wadding, gauze, bandages", [12.0]),
    "3006": ("Pharmaceutical preparations", [12.0]),
    "9018": ("Medical instruments and appliances", [12.0]),

    # Stationery & Office
    "4802": ("Uncoated paper for writing", [12.0]),
    "4820": ("Registers, notebooks, diaries", [12.0]),
    "9608": ("Ball point pens, markers", [18.0]),

    # Construction
    "2523": ("Portland cement", [28.0]),
    "7213": ("Hot-rolled bars of iron or steel", [18.0]),
    "6802": ("Worked stone and articles thereof", [12.0, 18.0]),

    # Furniture
    "9401": ("Seats and parts thereof", [18.0]),
    "9403": ("Other furniture and parts thereof", [18.0]),

    # IT Services (SAC codes)
    "9983": ("Other professional, technical services", [18.0]),
    "9971": ("Financial and related services", [18.0]),
    "9973": ("Leasing or rental services", [18.0]),
    "9981": ("Research and development services", [18.0]),
    "9982": ("Legal and accounting services", [18.0]),
    "9984": ("Telecommunications services", [18.0]),
    "9985": ("Support services", [18.0]),
    "9986": ("Support services to agriculture", [0.0, 5.0]),
    "9987": ("Maintenance and repair services", [18.0]),
    "9988": ("Manufacturing services on physical inputs", [18.0]),
    "9961": ("Transport of goods", [5.0, 12.0]),
    "9962": ("Transport of passengers", [5.0, 12.0]),
    "9963": ("Accommodation and food services", [5.0, 12.0, 18.0]),
    "9964": ("Passenger transport services", [5.0]),
    "9972": ("Real estate services", [12.0, 18.0]),
    "9991": ("Public administration services", [0.0]),
    "9992": ("Education services", [0.0]),
    "9993": ("Human health and social care", [0.0]),
    "9995": ("Services of households as employers", [0.0]),
    "9996": ("Services of extraterritorial organizations", [0.0]),
    "9997": ("Other services (hairdressing, spa, gym)", [18.0]),
}

# Indian state codes
STATE_CODES: dict[str, str] = {
    "01": "Jammu & Kashmir",
    "02": "Himachal Pradesh",
    "03": "Punjab",
    "04": "Chandigarh",
    "05": "Uttarakhand",
    "06": "Haryana",
    "07": "Delhi",
    "08": "Rajasthan",
    "09": "Uttar Pradesh",
    "10": "Bihar",
    "11": "Sikkim",
    "12": "Arunachal Pradesh",
    "13": "Nagaland",
    "14": "Manipur",
    "15": "Mizoram",
    "16": "Tripura",
    "17": "Meghalaya",
    "18": "Assam",
    "19": "West Bengal",
    "20": "Jharkhand",
    "21": "Odisha",
    "22": "Chhattisgarh",
    "23": "Madhya Pradesh",
    "24": "Gujarat",
    "25": "Daman & Diu",
    "26": "Dadra & Nagar Haveli",
    "27": "Maharashtra",
    "28": "Andhra Pradesh",
    "29": "Karnataka",
    "30": "Goa",
    "31": "Lakshadweep",
    "32": "Kerala",
    "33": "Tamil Nadu",
    "34": "Puducherry",
    "35": "Andaman & Nicobar Islands",
    "36": "Telangana",
    "37": "Andhra Pradesh (new)",
    "38": "Ladakh",
}

# Valid GST tax rates
VALID_TAX_RATES = [0.0, 5.0, 12.0, 18.0, 28.0]

# E-way bill threshold (in INR)
EWAY_BILL_THRESHOLD = 50000.0

# Composition scheme turnover limit (in INR)
COMPOSITION_SCHEME_LIMIT = 15000000.0  # 1.5 crore

# Reverse charge applicable categories (simplified)
REVERSE_CHARGE_SERVICES = {
    "9971",  # Financial services from non-registered
    "9973",  # Rental from non-registered
    "9985",  # Support services
    "9961",  # Goods transport agency
    "9962",  # Transport
}
