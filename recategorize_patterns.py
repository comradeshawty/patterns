import pandas as pd
from collections import Counter
from shapely.geometry import Polygon,Point
import json
from shapely.wkt import loads
pd.set_option('display.max_columns', None)
import geopandas as gpd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns
#from rapidfuzz import fuzz
from scipy.spatial import cKDTree
import ast
from ast import literal_eval
SUB_CATEGORY_MAPPING={'Restaurants':['Full-Service Restaurants','Casino Hotels','Limited-Service Restaurants'],
                      'Coffee Shops, Snacks & Bakeries':['Snack and Nonalcoholic Beverage Bars','Bakeries and Tortilla Manufacturing','Confectionery and Nut Stores','Baked Goods Stores'],
                      'Retail for Basic Necessities': ['All Other Health and Personal Care Stores','All Other Specialty Food Stores','Meat Markets','Fruit and Vegetable Markets','Fish and Seafood Markets',
                                                       'Supermarkets and Other Grocery (except Convenience) Stores','Gasoline Stations',
                                                       'Pharmacies and Drug Stores','Optical Goods Stores','Warehouse Clubs and Supercenters','Department Stores','Fuel Dealers','General Merchandise Stores, including Warehouse Clubs and Supercenters','All Other General Merchandise Stores'],
                      'Sports and Exercise':['Motorcycle, Bicycle, and Parts Manufacturing','Fitness and Recreational Sports Centers','Golf Courses and Country Clubs','Racetracks',],
                      'Healthcare':['General Medical and Surgical Hospitals','Offices of Other Health Practitioners', 'Offices of Mental Health Practitioners (except Physicians)',
                                  'Offices of Physicians','Blood and Organ Banks','Outpatient Care Centers','Offices of Physicians (except Mental Health Specialists)',
                                  'Offices of Dentists','Specialty (except Psychiatric and Substance Abuse) Hospitals','Medical and Diagnostic Laboratories',
                                  'Other Ambulatory Health Care Services'],
                      'Social Support':['Psychiatric and Substance Abuse Hospitals','Nursing Care Facilities (Skilled Nursing Facilities)','Other Information Services','Child Day Care Services',
                                     'Nursing and Residential Care Facilities','Community Food Services','Community Food and Housing, and Emergency and Other Relief Services',
                                     'Continuing Care Retirement Communities and Assisted Living Facilities for the Elderly', 'Home Health Care Services',
                                     'Individual and Family Services', 'Services for the Elderly and Persons with Disabilities','Administration of Public Health Programs.',
                                     'Social Assistance', 'Outpatient Mental Health and Substance Abuse Centers', 'Child and Youth Services',
                                     'Administration of Housing Programs, Urban Planning, and Community Development','Administration of Human Resource Programs'],
                      'Religious Organizations': ['Religious Organizations'],
                      'School':['Sports and Recreation Instruction','Elementary and Secondary Schools','Other Schools and Instruction','All Other Amusement and Recreation Industries'],
                      'Financial, Legal, Real Estate and Insurance Services':['Tax Preparation Services','Other Accounting Services','Financial Advice','Consumer Lending','Investment Advice','Accounting, Tax Preparation, Bookkeeping, and Payroll Services','Miscellaneous Financial Investment Activities','Credit Unions','All Other Nondepository Credit Intermediation','Other Activities Related to Credit Intermediation','Offices of Notaries','Commercial Banking','Offices of Lawyers','Insurance Agencies and Brokerages','Direct Life Insurance Carriers',
                                                                                'Other Direct Insurance (except Life, Health, and Medical) Carriers','Mortgage and Nonmortgage Loan Brokers','Tax Preparation Services','Other Accounting Services','Offices of Real Estate Agents and Brokers','Residential Property Managers'],
                      'City/Outdoors':['Cemeteries and Crematories','Justice, Public Order, and Safety Activities','RV (Recreational Vehicle) Parks and Recreational Camps','Nature Parks and Other Similar Institutions',
                                    'Executive, Legislative, and Other General Government Support','Administration of Economic Programs','Business Associations','Civic and Social Organizations','Courts','Fire Protection','Other Social Advocacy Organizations','Regulation and Administration of Transportation Programs','Police Protection','Cemeteries and Crematories','Correctional Institutions','Other General Government Support',
                                    'Civic and Social Organizations','Social Advocacy Organizations','Civic, Social, and Fraternal Associations','Grantmaking and Giving Services'],
                      'College':['Colleges, Universities, and Professional Schools','Junior Colleges','Technical and Trade Schools'],
                      'Arts and Culture':['Sound Recording Studios','Dance Companies','Photography Studios, Portrait','Motion Picture and Video Industries',
                                          'Scenic and Sightseeing Transportation','Zoos and Botanical Gardens','Art Dealers','Sound Recording Industries','Theater Companies and Dinner Theaters',
                                          'Radio and Television Broadcasting','Museums','Historical Sites','Performing Arts Companies'],
                      'Entertainment':['Promoters of Performing Arts, Sports, and Similar Events with Facilities','Motion Picture Theaters (except Drive-Ins)','Bowling Centers','Wineries','Amusement Parks and Arcades','Beverage Manufacturing',
                                    'Spectator Sports','Other Amusement and Recreation Industries','Casinos (except Casino Hotels)', 'Other Gambling Industries','Drinking Places (Alcoholic Beverages)'],
                      'Discretionary Retail':['General Rental Centers','Appliance Repair and Maintenance','Lawn and Garden Equipment and Supplies Stores','Home Furnishings Stores','Home Furnishings', 'Other Clothing Stores','Food (Health) Supplement Stores',
                                            'Used Merchandise Stores', 'Book Stores',
                                            'Cosmetics, Beauty Supplies, and Perfume Stores','Sporting Goods Stores', 'Family Clothing Stores',"Women's Clothing Stores",
                                            'All Other Miscellaneous Store Retailers (except Tobacco Stores)','Hobby, Toy, and Game Stores','Recreational Goods Rental','Video Tape and Disc Rental','Malls','Shopping Centers',
                                            'Photofinishing Laboratories (except One-Hour)','Automobile Dealers','Other Motor Vehicle Dealers','Automotive Parts and Accessories Stores','Automotive Equipment Rental and Leasing',
                                              'Automotive Parts, Accessories, and Tire Stores','Automotive Repair and Maintenance','Convenience Stores','Furniture Stores','Tobacco Stores', 'Beer, Wine, and Liquor Stores',
                                            'Sewing, Needlework, and Piece Goods Stores','Clothing Stores','Gift, Novelty, and Souvenir Stores', 'Electronics Stores','Jewelry Stores', 'Shoe Stores', 'Pet and Pet Supplies Stores',
                                            'All Other Home Furnishings Stores', 'Office Supplies and Stationery Stores', 'Floor Covering Stores','Musical Instrument and Supplies Stores',
                                              'Luggage and Leather Goods Stores',"Children's and Infants' Clothing Stores",
                                            'Nursery, Garden Center, and Farm Supply Stores','Household Appliance Stores', 'Clothing Accessories Stores', "Men's Clothing Stores",
                                            'Building Material and Supplies Dealers','Wired and Wireless Telecommunications Carriers','Other Wood Product Manufacturing'],
                      'Personal Services':['Photographic Services','Travel Arrangement and Reservation Services','Veterinary Services','Lessors of Miniwarehouses and Self-Storage Units','Consumer Goods Rental',
                                           'Offices of All Other Miscellaneous Health Practitioners','Death Care Services','Funeral Homes and Funeral Services',
                                        'Other Personal Services','Personal Care Services','Couriers and Express Delivery Services','Specialized Design Services',
                                        'Florists','Employment Services','Personal and Household Goods Repair and Maintenance',
                                        'All Other Consumer Goods Rental','Special Food Services','Drycleaning and Laundry Services',
                                           'Printing and Related Support Activities','Postal Service','Hotels (except Casino Hotels) and Motels','Electronic and Precision Equipment Repair and Maintenance'],
                   'Transportation':['Charter Bus Industry','Other Support Activities for Air Transportation',
                                     'Interurban and Rural Bus Transportation','Port and Harbor Operations',
                                     'Bus Rental','Other Transit and Ground Passenger Transportation','Taxi and Limousine Service','Passenger Car Rental','Motor Vehicle Towing','Parking Lots and Garages',
                                     'Transit and Ground Passenger Transportation','Urban Transit Systems','Rail Transportation'],
                      'College':['Colleges, Universities, and Professional Schools','Junior Colleges','Technical and Trade Schools'],
'Work':['Services to Buildings and Dwellings','Management, Scientific, and Technical Consulting Services','Management of Companies and Enterprises','Architectural Services','Advertising, Public Relations, and Related Services','Other Support Services','Newspaper, Periodical, Book, and Directory Publishers','Offices of Other Holding Companies','Lessors of Residential Buildings and Dwellings',
                      'Office Park','Lessors of Nonresidential Buildings (except Miniwarehouses)', 'Corporate, Subsidiary, and Regional Managing Offices','Lessors of Other Real Estate Property','Administrative Management and General Management Consulting Services','Newspaper, Periodical, Book, and Directory Publishers',
                      'Lessors of Nonresidential Buildings (except Miniwarehouses','Architectural, Engineering, and Related Services','Other Services to Buildings and Dwellings']
                      }

sub_categories_to_pretty_names={'Restaurants':{'Fast Food':['Limited-Service Restaurants'],
                                               'Restaurants':['Full Service Restaurants','Casino Hotels']
                                               },
                              'Sports and Exercise':{'Sports Complex':['Promoters of Performing Arts, Sports, and Similar Events with Facilities'],
                                                       'Golf Courses and Country Clubs':['Golf Courses and Country Clubs'],
                                                       'Racetracks':['Racetracks'],
                                                        'Gyms and Fitness Centers': ['Motorcycle, Bicycle, and Parts Manufacturing','Fitness and Recreational Sports Centers']},
                                'Coffee Shops, Snacks & Bakeries':{'Coffee Shop':['Starbucks',"Ohenry's Coffees", 'Costa Coffee','Revelator Coffee'],
                                                                   'Donuts':["Dunkin'",'Krispy Kreme Doughnuts','Shipley Donuts','Daylight Donuts'],
                                                                   'Bakery':['Cinnaholic','Insomnia Cookies','Great American Cookies''Nothing Bundt Cakes'],
                                                                   'Ice Cream & Frozen Yogurt':['Cold Stone Creamery', "Freddy's Frozen Custard",'Yogurt Mountain', 'Baskin Robbins', 'TCBY', 'Orange Julius',
                                                                                     'Marble Slab Creamery',"Bruster's Ice Cream",],
                                                                   'Smoothie & Juice Bar':['Tropical Smoothie Café','Clean Juice','Jamba','Planet Smoothie']},
                                'Entertainment':{'Movie Theater':['Motion Picture Theaters (except Drive-Ins)'],
                                                 'Sports Stadium':['Sports Teams and Clubs','Other Spectator Sports'],
                                                 'Entertainment Venue':['Promoters of Performing Arts, Sports, and Similar Events with Facilities'],
                                                 'Bars':['Drinking Places (Alcoholic Beverages)'],
                                                 'Bowling Alley':['Bowling Centers'],
                                                 'Amusement Parks':['Amusement and Theme Parks'],
                                                 'Brewery':['Breweries','Wineries'],
                                                 'Casino':['Casinos (except Casino Hotels)'],
                                                 'Arcade':['Amusement Arcades']},
                                'Work':{'Workplace':['Services to Buildings and Dwellings',
                                        'Management, Scientific, and Technical Consulting Services',
                                        'Advertising, Public Relations, and Related Services',
                                        'Architectural, Engineering, and Related Services',
                                        'Corporate, Subsidiary, and Regional Managing Offices',
                                        'Offices of Other Holding Companies',
                                        'Newspaper, Periodical, Book, and Directory Publishers',
                                        'Janitorial Services', 'Architectural Services',
                                        'Lessors of Nonresidential Buildings (except Miniwarehouses)',
                                        'Other Support Services',
                                        'Other Services to Buildings and Dwellings']},
                                'Arts and Culture':{'Art Gallery':['Art Dealers'],
                                                    'Fine Arts Schools':['Fine Arts Schools'],
                                                    'TV Station':['Radio and Television Broadcasting'],
                                                    'Historical Landmarks':['Historical Sites'],
                                                    'Museum':['Museums'],
                                                    'Dance Company':['Dance Companies'],
                                                    'Theater':['Theater Companies and Dinner Theaters'],
                                                    'Recording Studio':['Sound Recording Studios'],
                                                    'Botanical Garden':['Zoos and Botanical Gardens'],
                                                    "Movie Production Studio":['Motion Picture and Video Production']},
                                'Transportation':{'Parking Lots and Garages':['Parking Lots and Garages'],
                                                  'Intercity Bus':['Interurban and Rural Bus Transportation'],
                                                  'Charter Bus':['Charter Bus Industry'],
                                                  'Car Rental':['Passenger Car Rental'],
                                                  'Marina':['Port and Harbor Operations'],
                                                  'Airport Shuttle':['All Other Transit and Ground Passenger Transportation','Transit and Ground Passenger Transportation','Other Support Activities for Air Transportation'],
                                                  'Towing Company':['Motor Vehicle Towing'],
                                                  'Taxi & Limo':['Limousine Service','Taxi Service']},
                                'College':{'University':['Colleges, Universities, and Professional Schools'],
                                            'Community College':['Junior Colleges'],
                                            'Trade School':['Technical and Trade Schools'],
                                            'Cosmetology School':['Cosmetology and Barber Schools']},
                                'Retail for Basic Necessities':{'Grocery Store':['Supermarkets and Other Grocery (except Convenience) Stores'],
                                                                 'Fresh Food Market':['Fish and Seafood Markets', 'Fruit and Vegetable Markets','Meat Markets', 'All Other Specialty Food Stores'],
                                                                'Optical Goods Stores':['Optical Goods Stores'],
                                                                'Warehouse Clubs and Supercenters':['Warehouse Clubs and Supercenters','All Other General Merchandise Stores'],
                                                                'Gasoline Stations':['Gasoline Stations with Convenience Stores','Fuel Dealers'],
                                                                'Department Stores':['Department Stores'],
                                                                'Pharmacy':['Pharmacies and Drug Stores'],
                                                                'Medical Aids Store':['All Other Health and Personal Care Stores']},
                                'Personal Services':{'Beauty Salon':['Beauty Salons'],
                                                     'Nail Salon':['Nail Salons'],
                                                     'Spa':['Other Personal Care Services','Hair, Nail, and Skin Care Services','Personal Care Services', 'Other Personal Services'],
                                                     'Hotel':['Hotels (except Casino Hotels) and Motels'],
                                                     'Florist':['Florists'],
                                                     'Barber Shops':['Barber Shops'],
                                                     'Drycleaners':['Drycleaning and Laundry Services (except Coin-Operated)'],
                                                     'Event Center':['All Other Consumer Goods Rental'],
                                                     'Caterers':['Caterers'],
                                                     'Veterinarian':['Veterinary Services'],
                                                     'Electronics Repair':['Consumer Electronics Repair and Maintenance'],
                                                     'Household Goods Repair':['Home and Garden Equipment Repair and Maintenance','Reupholstery and Furniture Repair','Other Personal and Household Goods Repair and Maintenance','Footwear and Leather Goods Repair'],
                                                     'Post Office':['Postal Service'],
                                                     'Employment Placement Agency':['Temporary Help Services','Employment Placement Agencies'],
                                                     'Pet Care':['Pet Care (except Veterinary) Services'],
                                                     'Travel Agency':['Travel Arrangement and Reservation Services'],
                                                     'Graphic Designer':['Specialized Design Services'],
                                                     'Weight Loss Center':['Diet and Weight Reducing Centers'],
                                                     'Print & Ship Center':['Couriers and Express Delivery Services','Printing and Related Support Activities','Commercial Screen Printing'],
                                                     'Alternative Medicine':['Offices of All Other Miscellaneous Health Practitioners'],
                                                     'Photographer':['Photographic Services'],
                                                     'Storage Unit':['Lessors of Miniwarehouses and Self-Storage Units'],
                                                     'Funeral Home':['Death Care Services','Funeral Homes and Funeral Services']},
                                'Discretionary Retail':{'Mall':['Malls'],
                                                        'Shopping Center':['Shopping Centers'],
                                                        'Auto Body Shop':['All Other Automotive Repair and Maintenance','General Automotive Repair','Automotive Body, Paint, and Interior Repair and Maintenance','Automotive Glass Replacement Shops',
                                                                          'Automotive Transmission Repair','Other Automotive Mechanical and Electrical Repair and Maintenance'],
                                                        'Photo Prints Center':['Photofinishing Laboratories (except One-Hour)'],
                                                        'Convenience Store': ['Convenience Stores'],
                                                        'Hardware Store':['Hardware Stores'],
                                                        'Sporting Goods and Outdoor Gear':['Sporting Goods Stores'],
                                                        'Clothing Accessories Stores':['Clothing Accessories Stores'],
                                                        'Car Dealership':['New Car Dealers','Automobile Dealers','Used Car Dealers'],
                                                        'Car Parts Store':['Automotive Parts and Accessories Stores'],
                                                        'Tire Shop':['Tire Dealers'],
                                                        'Women\'s Clothing Store':["Women's Clothing Stores"],
                                                        'Men\'s Clothing Store':["Men's Clothing Stores",'Clothing Stores'],
                                                        'Shoe Store':['Shoe Stores'],
                                                        'Video Rental Store':['Video Tape and Disc Rental'],
                                                        'Arts & Crafts Store':['Sewing, Needlework, and Piece Goods Stores'],
                                                        'Farm and Garden Store':['Lawn and Garden Equipment and Supplies Stores','Nursery, Garden Center, and Farm Supply Stores'],
                                                        'Family and Children\'s Clothing Store':["Family Clothing Stores","Children's and Infants' Clothing Stores"],
                                                        'Other Motor Vehicle Dealers':['Boat Dealers','Motorcycle, ATV, and All Other Motor Vehicle Dealers','Truck, Utility Trailer, and RV (Recreational Vehicle) Rental and Leasing'],
                                                        'Car Wash':['Car Washes'],
                                                        'Used Merchandise Store':['Used Merchandise Stores'],
                                                        'Office Supplies and Stationery Store':['Office Supplies and Stationery Stores'],
                                                        'Bookstore':['Book Stores and News Dealers'],
                                                        'Furniture and Home Decor Store':['Floor Covering Stores','Home Furnishings Stores','Furniture Stores','All Other Home Furnishings Stores'],
                                                        'Jewelry Store':['Jewelry Stores'],
                                                        'Luggage, and Leather Goods Stores':['Luggage, and Leather Goods Stores','Luggage and Leather Goods Stores'],
                                                        'Gift & Souvenir Shop':['Gift, Novelty, and Souvenir Stores'],
                                                        'Pet Supply Store':['Pet and Pet Supplies Stores'],
                                                        'Cell Phone Provider':['Wired and Wireless Telecommunications Carriers'],
                                                        'Hobby, Toy, and Game Store':['Hobby, Toy, and Game Stores'],
                                                        'Home Improvement Store':['Building Material and Supplies Dealers','Other Building Material Dealers','Home Centers','Paint and Wallpaper Stores','Other Wood Product Manufacturing'],
                                                        'Household Appliance Store':['Household Appliance Stores','Appliance Repair and Maintenance'],
                                                        'Cosmetics/Fragrance Store':['Cosmetics, Beauty Supplies, and Perfume Stores'],
                                                        'Health Supplement Store':['Food (Health) Supplement Stores'],
                                                        'Smoke Shop':['Tobacco Stores'],
                                                        'Rental':['General Rental Centers','All Other Consumer Goods Rental'],
                                                        'Liquor Store':['Beer, Wine, and Liquor Stores'],
                                                        'Other Miscellaneous Store Retailers':['All Other Miscellaneous Store Retailers (except Tobacco Stores)']},
                                'Financial, Legal, Real Estate and Insurance Services':{'Real Estate Agents':['Offices of Real Estate Agents and Brokers','Residential Property Managers'],
                                                            'Mortgage Brokers':['Mortgage and Nonmortgage Loan Brokers'],
                                                            'Insurance Agency':['Insurance Agencies and Brokerages','Direct Life Insurance Carriers',
                                                                                'Other Direct Insurance (except Life, Health, and Medical) Carriers'],
                                                            'Lawyers':['Offices of Lawyers'],
                                                            'Bank':['Commercial Banking'],
                                                            'Notary':['Offices of Notaries'],
                                                            'Credit Unions':['Credit Unions','All Other Nondepository Credit Intermediation','Other Activities Related to Credit Intermediation',],
                                                            'Financial Advisors':['Financial Advice','Investment Advice','Miscellaneous Financial Investment Activities'],
                                                            'Consumer Lending':['Consumer Lending'],
                                                            'Accoutants':['Tax Preparation Services','Other Accounting Services']},
                                'Healthcare':{'Physical Therapist':['Offices of Physical, Occupational and Speech Therapists, and Audiologists'],
                                              'Kidney Dialysis Center':['Kidney Dialysis Centers'],'Medical Laboratories':['Medical Laboratories'],
                                              'Family Planning Center':['Family Planning Centers'],'Blood Bank':['Blood and Organ Banks'],
                                              'Psychiatrist\'s Office':['Offices of Physicians, Mental Health Specialists','Offices of Physicians, Mental Health Specialists'],
                                              'Dentist\'s Office':['Offices of Dentists','Teeth Whitening'],
                                              'Doctor\'s Office':['Offices of Physicians (except Mental Health Specialists)','Specialty (except Psychiatric and Substance Abuse) Hospitals'],
                                              'Hospital':['General Medical and Surgical Hospitals',],
                                              'Therapist':['Offices of Mental Health Practitioners (except Physicians)'],
                                              'Urgent Care':['Freestanding Ambulatory Surgical and Emergency Centers','Urgent Care','Urgent Care,Walk-in Clinic'],
                                              'Psychiatric and Substance Abuse Hospital':['Psychiatric and Substance Abuse Hospitals','Addiction Treatment'],
                                              'Optometrist':['Offices of Optometrists'],
                                              'Speech Therapist':['Speech Therapist'],
                                              'Sleep Clinic':['All Other Outpatient Care Centers'],
                                              'Chiropractor\'s Office':['Offices of Chiropractors']},
                                'Religious Organizations':{'Church':['Churches'],
                                                            'Mosque':['Mosque'],
                                                            'Buddhist Temple':['Buddhist'],
                                                            'Hindu Temple':['Hindu Temple'],
                                                            'Church Daycare':['Child Care']},
                                'Social Support':{'Home Health Care':['Home Health Care Services'],
                                                  'Community Housing Services':['Community Housing Services'],
                                                  'Daycare Center':['Child Day Care Services','Child Care'],
                                                  'Disability Services':['Services for the Elderly and Persons with Disabilities'],
                                                  'Library':['Libraries and Archives'],
                                                  'Community Center':['Other Individual and Family Services'],
                                                  'Social Security Office':['Administration of Human Resource Programs (except Education, Public Health, and Veterans Affairs Programs)'],
                                                  'Nursing Care Facility':['Nursing Care Facilities (Skilled Nursing Facilities)'],
                                                  'Child and Youth Services':['Child and Youth Services'],
                                                  'Assisted Living Facilities for the Elderly':['Assisted Living Facilities for the Elderly'],
                                                  'Substance Abuse Treatment Center':['Outpatient Mental Health and Substance Abuse Centers','New Season'],
                                                  'Temporary Shelters':['Temporary Shelters'],
                                                  'Food Bank':['Community Food Services']},
                                'School':{'Exam Prep & Tutoring Center':['Exam Preparation and Tutoring',],
                                          'Sports and Recreation Instruction':['Sports and Recreation Instruction'],
                                          'English Language Schools':['Language Schools'],
                                          'Dance School':['Fine Arts Schools'],
                                          'School':['Elementary and Secondary Schools'],
                                          'CPR & First Aid Classes':['All Other Miscellaneous Schools and Instruction'],
                                          'Summer Camp':['Summer Camp'],
                                          'Driving School':['Automobile Driving Schools'],
                                          'Art Classes':['Art Classes','Painting with a Twist']},
                                'City/Outdoors':{'Prison':['Correctional Institutions'],
                                                'Cemetery':['Cemeteries and Crematories'],
                                                'Police Station':['Police Protection'],
                                                'DMV':['Regulation and Administration of Transportation Programs','Administration of Economic Programs'],
                                                'Fire Station':['Fire Protection',],
                                                'Rotary Club':['Other Social Advocacy Organizations'],
                                                'Courthouse':['Courts'],
                                                'Park':['Nature Parks and Other Similar Institutions'],
                                                'Social Organization':['Civic and Social Organizations'],
                                                'Chamber of Commerce':['Business Associations'],
                                                'Nonprofit':['Voluntary Health Organizations'],
                                                 'RV Camp':['RV (Recreational Vehicle) Parks and Recreational Camps'],
                                                'Government Offices':['Executive, Legislative, and Other General Government Support','Other General Government Support']}}
def remove_nearby_duplicate_offices(mp_gdf, placekeys_to_drop_path, distance_threshold=20, fuzz_threshold=65):

    try:
        placekeys_to_drop = pd.read_csv(placekeys_to_drop_path)
    except FileNotFoundError:
        placekeys_to_drop = pd.DataFrame(columns=['PLACEKEY'])

    excluded_brands = {'Walmart', 'Winn Dixie', 'Walgreens', 'CVS','Publix'}
    excluded_categories = {'Child Day Care Services', 'Elementary and Secondary Schools','Child and Youth Services'}
    mp_filtered = mp_gdf[(~mp_gdf['BRANDS'].isin(excluded_brands)) &(~mp_gdf['LOCATION_NAME'].str.contains("Emergency", case=False, na=False)) &(~mp_gdf['LOCATION_NAME'].str.contains("Walmart|Winn Dixie|Walgreens|CVS|Publix", case=False, na=False)) &(~mp_gdf['TOP_CATEGORY'].isin(excluded_categories))].copy()
    mp_filtered = gpd.GeoDataFrame(mp_filtered.copy(), geometry=gpd.points_from_xy(mp_filtered.LONGITUDE, mp_filtered.LATITUDE), crs="EPSG:4326").to_crs(epsg=32616)
    mp_filtered=mp_filtered[mp_filtered['POLYGON_CLASS']=='SHARED_POLYGON']
    mp_filtered = mp_filtered.reset_index(drop=True)
    coords = np.array(list(zip(mp_filtered.geometry.x, mp_filtered.geometry.y)))
    tree = cKDTree(coords)

    to_remove = set()
    new_removed_rows = []
    new_placekeys_to_drop = []

    index_mapping = dict(zip(range(len(mp_filtered)), mp_filtered.index))

    for idx, coord in zip(mp_filtered.index, coords):
        if idx in to_remove:
            continue

        nearby_indices = [mp_filtered.index[i] for i in tree.query_ball_point(coord, distance_threshold)]
        current_name = mp_filtered.at[idx, 'LOCATION_NAME']
        current_address = mp_filtered.at[idx, 'address']

        duplicates = [idx]

        for i in nearby_indices:
            if i == idx or i in to_remove:
                continue

            nearby_name = mp_filtered.at[i, 'LOCATION_NAME']
            nearby_address = mp_filtered.at[i, 'address']

            name_similarity = fuzz.ratio(current_name, nearby_name)
            address_similarity = fuzz.ratio(current_address, nearby_address)

            if name_similarity >= fuzz_threshold and address_similarity >= fuzz_threshold:
                duplicates.append(i)

        if len(duplicates) > 1:
            duplicate_rows = mp_filtered.loc[duplicates]

            parent_rows = duplicate_rows[duplicate_rows['parent_flag'] == 1]

            if not parent_rows.empty:
                # Keep one of the parent_flag=1 rows with the highest visit count
                keep_idx = parent_rows['RAW_VISIT_COUNTS'].idxmax()
            else:
                # No parent_flag=1, so keep the row with the highest visit count
                keep_idx = duplicate_rows['RAW_VISIT_COUNTS'].idxmax()

            remove_idxs = [i for i in duplicates if i != keep_idx]

            # Handle case where neither removal condition is met (same visit counts & parent flags)
            if all(mp_filtered.loc[i, 'parent_flag'] == mp_filtered.loc[keep_idx, 'parent_flag'] for i in remove_idxs) and \
               all(mp_filtered.loc[i, 'RAW_VISIT_COUNTS'] == mp_filtered.loc[keep_idx, 'RAW_VISIT_COUNTS'] for i in remove_idxs):
                remove_idxs = remove_idxs[:1]

            # If both have parent_flag=1, update PARENT_PLACEKEY references
            if len(parent_rows) > 1:
                kept_placekey = mp_filtered.loc[keep_idx, 'PLACEKEY']
                for remove_idx in remove_idxs:
                    removed_placekey = mp_filtered.loc[remove_idx, 'PLACEKEY']
                    mp_gdf.loc[mp_gdf['PARENT_PLACEKEY'] == removed_placekey, 'PARENT_PLACEKEY'] = kept_placekey
                    print(f"**Updated PARENT_PLACEKEY:** Replaced {removed_placekey} → {kept_placekey}")

            print(f"\n**Keeping:** {mp_filtered.loc[keep_idx, 'PLACEKEY']} | {mp_filtered.loc[keep_idx, 'LOCATION_NAME']} | {mp_filtered.loc[keep_idx, 'address']}")
            print("**Removing:**")
            for i in remove_idxs:
                print(f"   - {mp_filtered.loc[i, 'PLACEKEY']} | {mp_filtered.loc[i, 'LOCATION_NAME']} | {mp_filtered.loc[i, 'address']} (Matched)")

            new_removed_rows.extend(mp_filtered.loc[remove_idxs].to_dict('records'))
            new_placekeys_to_drop.extend(mp_filtered.loc[remove_idxs, 'PLACEKEY'].tolist())
            to_remove.update(remove_idxs)

    new_removed_df = pd.DataFrame(new_removed_rows)
    mp_gdf_cleaned = mp_gdf.drop(index=mp_gdf.index.intersection(to_remove)).reset_index(drop=True)
    new_placekeys_df = pd.DataFrame({'PLACEKEY': new_placekeys_to_drop})
    placekeys_to_drop = pd.concat([placekeys_to_drop, new_placekeys_df], ignore_index=True).drop_duplicates()
    placekeys_to_drop.to_csv(placekeys_to_drop_path, index=False)
    print(f"\nAdded {len(new_placekeys_to_drop)} PLACEKEYs to `{placekeys_to_drop_path}`.")
    return mp_gdf_cleaned

def calculate_polygon_diameter(mp):
    diameters = []

    for geom in mp['POLYGON_WKT']:
        try:
            # Only apply loads() if it's still a string
            if isinstance(geom, str):
                polygon = loads(geom)  # Convert WKT string to Polygon
            else:# isinstance(geom, Polygon):
                polygon = geom  # Already a Polygon, no need to convert
            #else:
                #raise ValueError("Invalid data type for POLYGON_WKT")

            # Get bounding box (min_x, min_y, max_x, max_y)
            min_x, min_y, max_x, max_y = polygon.bounds

            # Compute geodesic distance between diagonal corners
            diameter = geodesic((min_y, min_x), (max_y, max_x)).meters
            diameters.append(diameter)

        except Exception as e:
            #print(f"Error processing geometry: {geom}, Error: {e}")
            diameters.append(None)

    mp['POLYGON_DIAMETER'] = diameters
    return mp

def remove_nearby_duplicate_offices_no_address(mp_gdf, placekeys_to_drop_path, fuzz_threshold=75):

    try:
        placekeys_to_drop = pd.read_csv(placekeys_to_drop_path)
    except FileNotFoundError:
        placekeys_to_drop = pd.DataFrame(columns=['PLACEKEY'])
    mp_gdf = calculate_polygon_diameter(mp_gdf)
    excluded_brands = {'Walmart', 'Winn Dixie', 'Walgreens', 'CVS','Publix',"Walmart Photo Center","Walmart Vision Center","Walmart Auto Care Center","Walmart Pharmacy","Woodforest National Bank","Jackson Hewitt Tax Service"}
    excluded_categories = {'Child Day Care Services', 'Elementary and Secondary Schools','Child and Youth Services'}
    mp_filtered = mp_gdf[
        (~mp_gdf['BRANDS'].isin(excluded_brands)) &
        (~mp_gdf['LOCATION_NAME'].str.contains("Emergency", case=False, na=False)) &
        #(~mp_gdf['LOCATION_NAME'].str.contains("Walmart|Winn Dixie|Walgreens|CVS|Publix", case=False, na=False)) &
        (~mp_gdf['TOP_CATEGORY'].isin(excluded_categories)) &
        (~mp_gdf['LOCATION_NAME'].isin(excluded_brands))  # **NEW EXCLUSION**
    ].copy()
    mp_filtered = gpd.GeoDataFrame(mp_filtered.copy(), geometry=gpd.points_from_xy(mp_filtered.LONGITUDE, mp_filtered.LATITUDE), crs="EPSG:4326").to_crs(epsg=32616)
    #mp_filtered=mp_filtered[mp_filtered['POLYGON_CLASS']=='SHARED_POLYGON']
    mp_filtered = mp_filtered.reset_index(drop=True)
    coords = np.array(list(zip(mp_filtered.geometry.x, mp_filtered.geometry.y)))
    tree = cKDTree(coords)

    to_remove = set()
    new_removed_rows = []
    new_placekeys_to_drop = []

    index_mapping = dict(zip(range(len(mp_filtered)), mp_filtered.index))

    for idx, coord in zip(mp_filtered.index, coords):
        if idx in to_remove:
            continue
        distance_threshold = mp_filtered.at[idx, 'POLYGON_DIAMETER']  # Use row-specific POLYGON_DIAMETER
        if pd.isna(distance_threshold):
            distance_threshold = 100  # Default fallback if missing

        nearby_indices = [mp_filtered.index[i] for i in tree.query_ball_point(coord, distance_threshold)]
        current_name = mp_filtered.at[idx, 'LOCATION_NAME']
        current_category = mp_filtered.at[idx, 'TOP_CATEGORY']
        duplicates = [idx]

        for i in nearby_indices:
            if i == idx or i in to_remove:
                continue

            nearby_name = mp_filtered.at[i, 'LOCATION_NAME']
            nearby_category = mp_filtered.at[i, 'TOP_CATEGORY']

            name_similarity = fuzz.ratio(current_name, nearby_name)
            #address_similarity = fuzz.ratio(current_address, nearby_address)
            if (
                ("Religious Organization" in {current_category, nearby_category}) and
                ("Child Day Care Services" in {current_category, nearby_category} or
                 any(keyword in current_name.lower() or keyword in nearby_name.lower() for keyword in ["childcare", "daycare", "child"]))
            ):
                continue  # Skip removing this pair

            if name_similarity >= fuzz_threshold:# and address_similarity >= fuzz_threshold:
                duplicates.append(i)

        if len(duplicates) > 1:
            duplicate_rows = mp_filtered.loc[duplicates]

            parent_rows = duplicate_rows[duplicate_rows['parent_flag'] == 1]

            if not parent_rows.empty:
                # Keep one of the parent_flag=1 rows with the highest visit count
                keep_idx = parent_rows['RAW_VISIT_COUNTS'].idxmax()
            else:
                # No parent_flag=1, so keep the row with the highest visit count
                keep_idx = duplicate_rows['RAW_VISIT_COUNTS'].idxmax()

            remove_idxs = [i for i in duplicates if i != keep_idx]

            # Handle case where neither removal condition is met (same visit counts & parent flags)
            if all(mp_filtered.loc[i, 'parent_flag'] == mp_filtered.loc[keep_idx, 'parent_flag'] for i in remove_idxs) and \
               all(mp_filtered.loc[i, 'RAW_VISIT_COUNTS'] == mp_filtered.loc[keep_idx, 'RAW_VISIT_COUNTS'] for i in remove_idxs):
                remove_idxs = remove_idxs[:1]

            # If both have parent_flag=1, update PARENT_PLACEKEY references
            if len(parent_rows) > 1:
                kept_placekey = mp_filtered.loc[keep_idx, 'PLACEKEY']
                for remove_idx in remove_idxs:
                    removed_placekey = mp_filtered.loc[remove_idx, 'PLACEKEY']
                    mp_gdf.loc[mp_gdf['PARENT_PLACEKEY'] == removed_placekey, 'PARENT_PLACEKEY'] = kept_placekey
                    print(f"**Updated PARENT_PLACEKEY:** Replaced {removed_placekey} → {kept_placekey}")

            print(f"\n**Keeping:** {mp_filtered.loc[keep_idx, 'PLACEKEY']} | {mp_filtered.loc[keep_idx, 'LOCATION_NAME']} | {mp_filtered.loc[keep_idx, 'address']}")
            print("**Removing:**")
            for i in remove_idxs:
                print(f"   - {mp_filtered.loc[i, 'PLACEKEY']} | {mp_filtered.loc[i, 'LOCATION_NAME']} | {mp_filtered.loc[i, 'address']} (Matched)")

            new_removed_rows.extend(mp_filtered.loc[remove_idxs].to_dict('records'))
            new_placekeys_to_drop.extend(mp_filtered.loc[remove_idxs, 'PLACEKEY'].tolist())
            to_remove.update(remove_idxs)

    new_removed_df = pd.DataFrame(new_removed_rows)
    mp_gdf_cleaned = mp_gdf.drop(index=mp_gdf.index.intersection(to_remove)).reset_index(drop=True)
    new_placekeys_df = pd.DataFrame({'PLACEKEY': new_placekeys_to_drop})
    placekeys_to_drop = pd.concat([placekeys_to_drop, new_placekeys_df], ignore_index=True).drop_duplicates()
    placekeys_to_drop.to_csv(placekeys_to_drop_path, index=False)
    print(f"\nAdded {len(new_placekeys_to_drop)} PLACEKEYs to `{placekeys_to_drop_path}`.")
    return mp_gdf_cleaned
def parent_childs(df_filtered):
    parent_placekeys_set = set(df_filtered["PARENT_PLACEKEY"].dropna().unique())
    parent_placekey_dfs = df_filtered.loc[df_filtered["PLACEKEY"].isin(parent_placekeys_set)].copy()
    parent_placekey_set = set(parent_placekey_dfs["PLACEKEY"])
    df_filtered = df_filtered.copy()
    df_filtered.loc[:, 'parent_flag'] = df_filtered['PLACEKEY'].apply(lambda pk: 1 if pk in parent_placekey_set else 0)
    parent_counts = (len(df_filtered.loc[df_filtered['parent_flag'] == 1]) / len(df_filtered)) * 100
    print(f"Percentage of parent placekeys: {parent_counts}")
    return df_filtered
def view_category(mp, category):
    return mp[mp['TOP_CATEGORY'].isin(category)]
def view_sub_category(mp, sub_category):
    return mp[mp['SUB_CATEGORY'].isin(sub_category)]
def view_brands(mp, brands):
    return mp[mp['BRANDS'].isin(brands)]
def view_catlabel(mp,catlabel):
  return mp[mp['three_cat_label'].isin(catlabel)]
def update_placekey_info(mp, placekeys, new_location_name=None,new_top_category=None, new_subcategory=None, new_naics_code=None, new_category_tags=None):
    if isinstance(placekeys, str):
        placekeys = [placekeys]

    mask = mp['PLACEKEY'].isin(placekeys)
    if not mask.any():
        print(f"No matching PLACEKEYs found in mp for {placekeys}. Skipping update.")
        return mp
    if new_location_name:
        mp.loc[mask, 'LOCATION_NAME'] = new_location_name
    if new_top_category:
        mp.loc[mask, 'TOP_CATEGORY'] = new_top_category
    if new_subcategory:
        mp.loc[mask, 'SUB_CATEGORY'] = new_subcategory
    if new_naics_code:
        mp.loc[mask, 'NAICS_CODE'] = new_naics_code
    if new_category_tags:
        mp.loc[mask, 'CATEGORY_TAGS'] = mp.loc[mask, 'CATEGORY_TAGS'].fillna('') + \
                                         (', ' if mp.loc[mask, 'CATEGORY_TAGS'].notna().all() else '') + \
                                         new_category_tags
    print(f"Updated {mask.sum()} rows for PLACEKEYs: {placekeys[:5]}{'...' if len(placekeys) > 5 else ''}")
    return mp

def update_legal_services(mp):
    mask = mp['LOCATION_NAME'].str.contains(r'Law|Atty|Attorney|Law Firm', case=False, na=False, regex=True)
    correct_classification = ((mp['TOP_CATEGORY'] == 'Legal Services') &(mp['SUB_CATEGORY'] == 'Offices of Lawyers') &(mp['NAICS_CODE'] == '541110'))
    mp.loc[mask & ~correct_classification, ['TOP_CATEGORY', 'SUB_CATEGORY', 'NAICS_CODE']] = ['Legal Services', 'Offices of Lawyers', '541110']
    return mp
def update_theater_companies(mp):
    mask = mp['LOCATION_NAME'].str.contains(r'Theater|Theatre', case=False, na=False)
    mp.loc[mask, ['TOP_CATEGORY', 'SUB_CATEGORY', 'NAICS_CODE']] = ['Performing Arts Companies', 'Theater Companies and Dinner Theaters', '711110']
    return mp
def update_non_court_services(mp):
    subcategory_mask = mp['SUB_CATEGORY'] == 'Courts'
    location_mask = ~mp['LOCATION_NAME'].str.contains(r'Court|Courthouse', case=False, na=False)
    target_rows = subcategory_mask & location_mask
    mp.loc[target_rows, ['TOP_CATEGORY', 'SUB_CATEGORY', 'NAICS_CODE']] = ['Legal Services', 'Offices of Lawyers', '541110']
    return mp

def remove_wholesalers(mp):
    mp=mp.copy()
    mask = mp['TOP_CATEGORY'].str.contains('Wholesalers', na=False) | \
           mp['SUB_CATEGORY'].str.contains('Wholesalers', na=False)
    mp = mp.loc[~mask].reset_index(drop=True)
    return mp

def update_religious_organizations(mp_gdf):
    religious_terms = ['Church', 'Temple', 'Synagogue', 'Mosque', 'Chapel', 'Cathedral','Basilica', 'Shrine', 'Monastery', 'Gurdwara', 'Tabernacle', 'Missionary','Worship Center', 'Bible Camp', 'Parish','Ministry','Ministries']
    religious_pattern = r'\b(?:' + '|'.join(religious_terms) + r')\b'
    religious_mask = mp_gdf['LOCATION_NAME'].str.contains(religious_pattern, flags=re.IGNORECASE, regex=True, na=False)
    mp_gdf.loc[religious_mask, ['TOP_CATEGORY', 'SUB_CATEGORY', 'NAICS_CODE']] = ['Religious Organizations', 'Religious Organizations', '813110']
    return mp_gdf

def update_real_estate_info(mp):
    mp['NAICS_CODE'] = mp['NAICS_CODE'].astype(str)
    mp.loc[mp['NAICS_CODE'] == '531190', 'NAICS_CODE'] = '531120'
    mp.loc[mp['SUB_CATEGORY'] == 'Lessors of Other Real Estate Property','SUB_CATEGORY'] = 'Lessors of Nonresidential Buildings (except Miniwarehouses)'
    mp.loc[mp['SUB_CATEGORY'].isin(['Malls', 'Shopping Centers']), 'NAICS_CODE'] = '531120'
    return mp

def preprocess_mp(mp):
  mp.drop_duplicates(subset='PLACEKEY', inplace=True, ignore_index=True)
  mp.dropna(subset='PLACEKEY', inplace=True, ignore_index=True)
  mp.dropna(subset='VISITOR_HOME_CBGS', inplace=True, ignore_index=True)
  mp.dropna(subset='RAW_VISIT_COUNTS', inplace=True, ignore_index=True)
  mp.dropna(subset='LOCATION_NAME', inplace=True, ignore_index=True)
  mp['TOP_CATEGORY']=mp['TOP_CATEGORY'].astype('str')
  mp['POSTAL_CODE']=mp['POSTAL_CODE'].astype('Int64').astype('str')
  mp['NAICS_CODE']=mp['NAICS_CODE'].astype('Int64').astype('str')
  mp['POI_CBG']=mp['POI_CBG'].astype('Int64').astype('str')
  mp.drop_duplicates(subset=['LOCATION_NAME','STREET_ADDRESS'], inplace=True, ignore_index=True)
  mp.loc[(mp['SUB_CATEGORY'] == 'Malls') & (mp['RAW_VISIT_COUNTS'] < 20000),'SUB_CATEGORY'] = 'Shopping Centers'
  categories_to_drop=['Household Appliance Manufacturing','Warehousing and Storage','Other Miscellaneous Manufacturing','General Warehousing and Storage','Machinery, Equipment, and Supplies Merchant Wholesalers',
  'Building Finishing Contractors','Building Equipment Contractors','Investigation and Security Services','Machinery, Equipment, and Supplies Merchant Wholesalers','Electrical Equipment Manufacturing',
                      'Residential Building Construction','Waste Treatment and Disposal','Waste Management and Remediation Services','Other Specialty Trade Contractors','Motor Vehicle Manufacturing','Miscellaneous Durable Goods Merchant Wholesalers',
                      'Steel Product Manufacturing from Purchased Steel','Glass and Glass Product Manufacturing','Professional and Commercial Equipment and Supplies Merchant Wholesalers','Securities and Commodity Contracts Intermediation and Brokerage',
  'Chemical and Allied Products Merchant Wholesalers','Commercial and Industrial Machinery and Equipment Rental and Leasing','Foundation, Structure, and Building Exterior Contractors','Freight Transportation Arrangement',
  'Lumber and Other Construction Materials Merchant Wholesalers','Specialized Freight Trucking','Business Support Services','Waste Management and Remediation Services','Glass and Glass Product Manufacturing','Data Processing, Hosting, and Related Services',
                      'Coating, Engraving, Heat Treating, and Allied Activities',
                      'Apparel Accessories and Other Apparel Manufacturing']
  sub_categories_to_remove=['Septic Tank and Related Services','Outdoor Power Equipment Stores',
                            'Cable and Other Subscription Programming','Refrigeration Equipment and Supplies Merchant Wholesalers',
                            'Packing and Crating','Other Electronic Parts and Equipment Merchant Wholesalers','Plumbing and Heating Equipment and Supplies (Hydronics) Merchant Wholesalers',
                            'All Other Support Services']
  mp = mp[~mp['TOP_CATEGORY'].isin(categories_to_drop)]
  mp = mp[~mp['SUB_CATEGORY'].isin(sub_categories_to_remove)]
  mp = mp.replace(r'\t', '', regex=True)
  mp = mp[~((mp['SUB_CATEGORY'] == 'Couriers and Express Delivery Services') & (mp['BRANDS'] != 'FedEx'))]
  mp=mp[mp['PLACEKEY']!='zzw-222@8gk-tv9-wrk']
  mp=mp[mp['IS_SYNTHETIC']==False]
  mp = mp.reset_index(drop=True)
  return mp
def process_pois_and_stops(mp, stops, radius=250):
    #stops = stops.loc[stops['TOP_CATEGORY'] == 'Urban Transit Systems'].copy()
    #stops.loc[:, 'LOCATION_NAME'] = stops['LOCATION_NAME'].str.replace(r'^Birmingham Jefferson County Transit Authority\s*', '', regex=True)
    stops.loc[:, 'geometry'] = stops.apply(lambda row: Point(row['stop_lon'], row['stop_lat']), axis=1)
    stops_gdf = gpd.GeoDataFrame(stops, geometry='geometry', crs="EPSG:4326").to_crs(epsg=32616)
    mp=mp.copy()
    mp.loc[:, 'geometry'] = mp.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)
    pois_gdf = gpd.GeoDataFrame(mp, geometry='geometry', crs="EPSG:4326").to_crs(epsg=32616)
    stop_coords = list(zip(stops_gdf.geometry.x, stops_gdf.geometry.y))
    poi_coords = list(zip(pois_gdf.geometry.x, pois_gdf.geometry.y))
    stop_tree = cKDTree(stop_coords)
    results = []
    for idx, poi_coord in enumerate(poi_coords):
        stop_indices = stop_tree.query_ball_point(poi_coord, radius)
        valid_distances = [np.sqrt((poi_coord[0] - stop_coords[i][0])**2 + (poi_coord[1] - stop_coords[i][1])**2) for i in stop_indices]
        valid_stop_indices = [i for i, dist in zip(stop_indices, valid_distances) if dist < radius]
        nearby_stop_names = [stops_gdf.iloc[i]['stop_name'] for i in valid_stop_indices]
        nearby_stop_ids = [stops_gdf.iloc[i]['stop_id'] for i in valid_stop_indices]
        results.append({
            'PLACEKEY': pois_gdf.iloc[idx]['PLACEKEY'],
            'nearby_stops': nearby_stop_names,
            'nearby_stop_ids':nearby_stop_ids,
            'nearby_stop_distances': valid_distances})
    nearby_pois = pd.DataFrame(results)
    nearby_pois = nearby_pois[(nearby_pois['nearby_stops'].apply(lambda x: len(x) > 0)) &
                              (nearby_pois['nearby_stop_distances'].apply(lambda x: len(x) > 0))].reset_index(drop=True)
    mp = mp.merge(nearby_pois, on="PLACEKEY", how="left")
    mp.loc[:, 'nearby_stops'] = mp['nearby_stops'].fillna('[]').apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    mp.loc[:, 'nearby_stop_distances'] = mp['nearby_stop_distances'].fillna('[]')
    mp.loc[:, 'num_nearby_stops'] = mp['nearby_stops'].apply(len)
    mp.drop_duplicates(subset="PLACEKEY", inplace=True, ignore_index=True)
    mp.dropna(subset="PLACEKEY", inplace=True, ignore_index=True)
    return mp
  
def extract_visit_counts_by_day(mp):
    def parse_popularity_by_day(value):
        if isinstance(value, str):
            try:
                parsed_dict = ast.literal_eval(value)  # Convert string to dictionary
                return parsed_dict if isinstance(parsed_dict, dict) else {}
            except (SyntaxError, ValueError):
                return {}
        return value if isinstance(value, dict) else {}
    mp['POPULARITY_BY_DAY'] = mp['POPULARITY_BY_DAY'].apply(parse_popularity_by_day)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in weekdays:
        mp[f'visit_count_{day.lower()}'] = mp['POPULARITY_BY_DAY'].apply(lambda x: x.get(day, 0))
    return mp

def drop_duplicates_with_priority(mp, save_path='/content/drive/MyDrive/data/removed_children.csv'):
    mp['VISITOR_HOME_CBGS_STR'] = mp['VISITOR_HOME_CBGS'].astype(str)
    mp_sorted = mp[mp['PARENT_PLACEKEY'].notna()].copy()
    mp_sorted = mp_sorted.sort_values(by=['PARENT_PLACEKEY', 'VISITOR_HOME_CBGS_STR', 'RAW_VISIT_COUNTS'], ascending=[True, True, False])
    cleaned_mp = mp_sorted.drop_duplicates(subset=['PARENT_PLACEKEY', 'VISITOR_HOME_CBGS'], keep='first')
    dropped_rows = mp_sorted[~mp_sorted.index.isin(cleaned_mp.index)]
    dropped_rows.to_csv(save_path, index=False)
    cleaned_mp = cleaned_mp.drop(columns=['VISITOR_HOME_CBGS_STR'])
    cleaned_mp.sort_values(by='RAW_VISIT_COUNTS', ascending=False, inplace=True)
    cleaned_mp.reset_index(drop=True, inplace=True)
    return cleaned_mp, dropped_rows

def three_cat_label(mp):
  mp['three_cat_label'] = mp.apply(lambda row: (
          row['TOP_CATEGORY']  # If both SUB_CATEGORY and CATEGORY_TAGS are NaN
          if pd.isna(row['SUB_CATEGORY']) and pd.isna(row['CATEGORY_TAGS'])
          else f"{row['TOP_CATEGORY']}-{row['SUB_CATEGORY']}"  # If only CATEGORY_TAGS is NaN
          if pd.isna(row['CATEGORY_TAGS'])
          else f"{row['TOP_CATEGORY']}-{row['SUB_CATEGORY']}-{row['CATEGORY_TAGS']}"),axis=1)
  return mp

def convert_placekey_to_stop(mp, placekey, stop_name):
  mp.loc[mp['PLACEKEY'] == placekey, ['LOCATION_NAME', 'TOP_CATEGORY', 'SUB_CATEGORY', 'NAICS_CODE','CATEGORY_TAGS']] = [stop_name,'Urban Transit Systems','Bus and Other Motor Vehicle Transit Systems','485113','Bus Station,Buses']
  return mp
  
def update_mp_from_w(mp, w, columns_to_update):
    w_lookup = {col: w.set_index("PLACEKEY")[col].to_dict() for col in columns_to_update}
    if "PLACEKEY" not in mp.columns:
        raise ValueError("PLACEKEY column is missing in mp")
    for col in columns_to_update:
        if col in mp.columns:
            w_values = mp["PLACEKEY"].map(w_lookup[col])
            mask = (w_values.notna()) & (mp[col] != w_values)
            mp.loc[mask, col] = w_values[mask]
    return mp

def merge_duplicate_pois(mp, save_path="/content/drive/MyDrive/data/removed_duplicate_pois.csv"):
    mp["POLYGON_ID"] = mp["PLACEKEY"].str.split("@").str[1]
    mp["VISITOR_HOME_CBGS_STR"] = mp["VISITOR_HOME_CBGS"].astype(str)
    mp=mp.sort_values(by='RAW_VISIT_COUNTS',ascending=False)
    grouped = mp.groupby(["POLYGON_ID", "VISITOR_HOME_CBGS_STR"])

    merged_rows = []
    removed_rows = []

    for (polygon_id, home_cbgs), group in grouped:
        if len(group) >= 2:  # Only process groups with duplicates

            # Step 2: Prioritize row where parent_flag = 1
            parent_rows = group[group["parent_flag"] == 1]

            if len(parent_rows) == 1:
                first_row = parent_rows.iloc[0].copy()  # Keep the parent row
            else:
                first_row = group.iloc[0].copy()  # If multiple or none are parents, keep the first row
            # Determine new LOCATION_NAME (Most common TOP_CATEGORY - Most common STREET ADDRESS)
            most_common_top_category = Counter(group["TOP_CATEGORY"]).most_common(1)[0][0]
            most_common_sub_category = Counter(group["SUB_CATEGORY"]).most_common(1)[0][0]
            most_common_tag_category = Counter(group["CATEGORY_TAGS"]).most_common(1)[0][0]

            most_common_address = Counter(group["STREET_ADDRESS"]).most_common(1)[0][0]
            first_row["LOCATION_NAME"] = f"{most_common_sub_category} - {most_common_address}"
            first_row["TOP_CATEGORY"]=most_common_top_category
            first_row["SUB_CATEGORY"]=most_common_sub_category
            first_row["CATEGORY_TAGS"]=most_common_tag_category
            # Keep only the first row, discard the rest
            merged_rows.append(first_row)
            removed_rows.append(group.iloc[1:])  # Store dropped rows

    # Create a DataFrame for removed rows and save them
    if removed_rows:
        removed_df = pd.concat(removed_rows)
        removed_df.to_csv(save_path, index=False)
    else:
        removed_df = pd.DataFrame()

    # Create final cleaned dataframe
    cleaned_mp = mp[~mp.index.isin(removed_df.index)]
    cleaned_mp = pd.concat([cleaned_mp, pd.DataFrame(merged_rows)], ignore_index=True)

    # Drop temporary columns
    cleaned_mp.drop(columns=["POLYGON_ID", "VISITOR_HOME_CBGS_STR"], inplace=True)

    return cleaned_mp, removed_df

def update_category(mp, placekeys, new_category):
    if isinstance(placekeys, str):
        placekeys = [placekeys]
    mp.loc[mp["PLACEKEY"].isin(placekeys), "place_category"] = new_category

    return mp
def assign_place_category_and_subcategory(mp, sub_category_mapping, sub_categories_to_pretty_names):
    # Step 1: Assign categories using the sub_category_mapping
    category_lookup = {
        subcategory: category
        for category, subcategories in sub_category_mapping.items()
        for subcategory in subcategories}

    mp["place_category"] = mp["SUB_CATEGORY"].map(category_lookup).fillna(mp["TOP_CATEGORY"].map(category_lookup)).fillna("Other")  # Default to "Other"
    def map_subcategory(row):
        top_category = row["SUB_CATEGORY"]
        place_category = row["place_category"]
        for main_category, subcategories in sub_categories_to_pretty_names.items():
            for subcategory, values in subcategories.items():
                if top_category in values:
                    return subcategory
        return f"Other {place_category}"  # If no match is found, return 'Other {place_category}'
    mp["place_subcategory"] = mp.apply(map_subcategory, axis=1)
    category_keywords = {
        "Schools": ["School", "Schools", "Academy", "Sch", "Montessori", "Summer Camp"],
        "City/Outdoors": ["Recreation Center", "City", "Playground", "Hiking", "Trail", "Courthouse"],
        "Arts and Culture": ["Mural", "Museum", "Artist",'Arts',"Cultural", "Dance", "Ballroom", "Exhibit"],
        "Entertainment": ["Trampoline Park", "Happy Hour", "Beer", "Beer Garden", "Mini Golf", "Topgolf",
                          "Pool and Billiards", "Axe Throwing", "Arcade", "Casino", "Go Kart", "Laser Tag",
                          "Escape Room", "Nightclub", "Comedy Club", "Event Space", "Speakeasy", "Theme Park",
                          "Water Park", "Winery", "Resort"],
        "Sports and Exercise": ["Tennis", "Gym", "Gymnastics", "Golf", "Yoga", "Rock Climbing", "Swimming",
                                "Baseball Field", "Athletics & Sports", "Basketball", "Climbing Gym",
                                "Hockey", "Skating Rink", "Soccer", "Boxing", "Squash", "Stable", "Volleyball"],
        "Work": ["Office Park", "Corporate Offices", "Corporate Office", "Business Center", "Conference",
                 "Coworking Space", "Meeting Room", "Industrial", "Non-Profit", "Tech Startup", "Warehouse"],
        "Personal Services": ["Barber", "Massage Therapy", "Spa", "Eyebrow", "Waxing", "Tattoo", "Medical Spa",
                              "Skin Care", "Contractors", "Handyman"],
        "Transportation": ["Greyhound", "Amtrak", "Airport", "Bus Station", "Train Station",
                           "Parking", "Taxi", "Terminal", "Travel", "Tunnel"],
        'Coffee Shops, Snacks & Bakeries':['Coffee','Bakery','Treats','Creamery','Smoothie','Donuts',"Jeni's Splendid Ice Creams",
                                           'Yogurt','Doughnuts','Tea','Teahouse','Ice Creams','Ice Cream','Crumbl Cookies','Frutta Bowls']
    }
    coffee_keywords=['Coffee','Bakery','Treats','Creamery','Smoothie','Donuts',"Jeni's Splendid Ice Creams",'Yogurt','Doughnuts','Tea','Teahouse','Ice Creams','Ice Cream','Crumbl Cookies','Frutta Bowls']
    arts_keywords=["Performing Arts","Mural"]
    # Step 3: Standardize LOCATION_NAME and CATEGORY_TAGS
    mp["LOCATION_NAME"] = mp["LOCATION_NAME"].str.strip()
    if "CATEGORY_TAGS" in mp.columns:
        mp["CATEGORY_TAGS"] = mp["CATEGORY_TAGS"].str.strip()

    # Step 4: Assign categories only if at least one match occurs
    for category, keywords in category_keywords.items():
        name_match = mp["LOCATION_NAME"].isin(keywords)  # Exact match for LOCATION_NAME

        if "CATEGORY_TAGS" in mp.columns:
            tag_match = mp["CATEGORY_TAGS"].isin(keywords)  # Exact match for CATEGORY_TAGS
        else:
            tag_match = False  # If CATEGORY_TAGS doesn't exist, skip this condition

        # Apply update only if at least one of the conditions is True
        update_mask = (name_match | tag_match) & (mp["place_category"] == "Other")
        mp.loc[update_mask, "place_category"] = category

    mp.loc[mp['LOCATION_NAME'].str.contains('Pharmacy', case=True, na=False), 'place_category'] = 'Retail for Basic Necessities'
    mp.loc[mp['LOCATION_NAME'].str.contains('Recreation Center', case=True, na=False), 'place_category'] = 'City/Outdoors'
    mp.loc[mp["LOCATION_NAME"].str.contains("|".join(coffee_keywords), case=True, na=False),'place_category']=='Coffee Shops, Snacks & Bakeries'
    mp.loc[mp["LOCATION_NAME"].str.contains("|".join(arts_keywords), case=True, na=False),'place_category']=='Arts and Culture'
    mp.loc[mp['PLACEKEY']=='227-222@8gk-td2-n5z','place_category']='City/Outdoors'
    mp.loc[mp['PLACEKEY']=='227-222@8gk-td2-n5z','place_subcategory']='Housing Authority'
    mp.loc[mp['PLACEKEY']=='zzw-223@8gk-tv9-qpv','place_category']='College'
    return mp

def assign_specific_subcategories(mp): 
    keyword_mappings = {
        'Coffee Shop': ['Starbucks',"Ohenry's Coffees", 'Costa Coffee','Revelator Coffee'],
        'Donuts': ["Dunkin'",'Krispy Kreme Doughnuts','Shipley Donuts','Daylight Donuts'],
        'Bakery': ['Cinnaholic','Insomnia Cookies','Great American Cookies', 'Nothing Bundt Cakes'],
        'Ice Cream & Frozen Yogurt': ['Cold Stone Creamery', "Freddy's Frozen Custard",'Yogurt Mountain', 
                                      'Baskin Robbins', 'TCBY', 'Orange Julius','Marble Slab Creamery',
                                      "Bruster's Ice Cream"],
        'Smoothie & Juice Bar': ['Tropical Smoothie Café','Clean Juice','Jamba','Planet Smoothie']
    }

    # Function to check if any of the keywords exist in the specified columns
    def match_subcategory(row):
        if row["place_category"] != "Coffee Shops, Snacks & Bakeries":
            return row["place_subcategory"]  # If not in the category, retain existing value
        
        for subcategory, keywords in keyword_mappings.items():
            if row["BRANDS"] in keywords or row["CATEGORY_TAGS"] in keywords or row["LOCATION_NAME"] in keywords:
                return subcategory
        return row["place_subcategory"]  # Keep existing value if no match

    # Apply the function to update place_subcategory only for relevant rows
    mp["place_subcategory"] = mp.apply(match_subcategory, axis=1)
    mp.loc[mp["place_category"].isin(["Work", "Other"]), "place_subcategory"] = "Work"

    return mp

