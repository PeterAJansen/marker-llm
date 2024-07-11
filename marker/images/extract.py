from marker.images.save import get_image_filename
from marker.pdf.images import render_bbox_image
from marker.schema.bbox import rescale_bbox
from marker.schema.block import find_insert_block, Span, Line
from marker.settings import settings


def find_image_blocks(page):
    image_blocks = []
    image_regions = [l.bbox for l in page.layout.bboxes if l.label in ["Figure", "Picture"]]
    image_regions = [rescale_bbox(page.layout.image_bbox, page.bbox, b) for b in image_regions]

    insert_points = {}
    for region_idx, region in enumerate(image_regions):
        for block_idx, block in enumerate(page.blocks):
            for line_idx, line in enumerate(block.lines):
                if line.intersection_pct(region) > settings.BBOX_INTERSECTION_THRESH:
                    line.spans = [] # We will remove this line from the block

                    if region_idx not in insert_points:
                        insert_points[region_idx] = (block_idx, line_idx)

    # Account for images with no detected lines
    for region_idx, region in enumerate(image_regions):
        if region_idx in insert_points:
            continue

        insert_points[region_idx] = (find_insert_block(page.blocks, region), 0)

    for region_idx, image_region in enumerate(image_regions):
        image_insert = insert_points[region_idx]
        image_blocks.append([image_insert[0], image_insert[1], image_region])

    return image_blocks


## PJ: OLD VERSIONS OF TABLE EXTRACTION

# def extract_page_images(page_obj, page, page_idx, raw_char_blocks_min):
#     if (page.images == None) or (type(page.images) != list):        ## PJ: Since the Tables are added into this images section, and tables are processed first, here it checks to make sure we don't overwrite any already-extracted tables
#         page.images = []
#     image_blocks = find_image_blocks(page)

#     for image_idx, (block_idx, line_idx, bbox) in enumerate(image_blocks):
#         if block_idx >= len(page.blocks):
#             block_idx = len(page.blocks) - 1
#         if block_idx < 0:
#             continue

#         block = page.blocks[block_idx]
#         image = render_bbox_image(page_obj, page, bbox)
#         image_filename = get_image_filename(page, image_idx)
#         image_markdown = f"\n\n![{image_filename}]({image_filename})\n\n"
#         image_span = Span(
#             bbox=bbox,
#             text=image_markdown,
#             font="Image",
#             rotation=0,
#             font_weight=0,
#             font_size=0,
#             image=True,
#             span_id=f"image_{image_idx}"
#         )

#         # Sometimes, the block has zero lines
#         if len(block.lines) > line_idx:
#             block.lines[line_idx].spans.append(image_span)
#         else:
#             line = Line(
#                 bbox=bbox,
#                 spans=[image_span]
#             )
#             block.lines.append(line)
#         page.images.append(image)

#         ## PJ: Try to extract text from the image
#         image_text = getTextFromBBox(page_idx, bbox, raw_char_blocks_min)
#         if image_text is not None:
#             print("IMAGE TEXT:")
#             print("PAGE:", page_idx)
#             print("BBOX:", bbox)
#             # json
#             import json
#             print(json.dumps(image_text, indent=4))


# def extract_images(doc, pages, raw_char_blocks_min):
#     for page_idx, page in enumerate(pages):
#         page_obj = doc[page_idx]
#         extract_page_images(page_obj, page, page_idx, raw_char_blocks_min)


## PJ: NEW VERSIONS OF TABLE EXTRACTION, THAT TRY TO DO TABLES + IMAGES AT ONCE.

def extract_page_images(page_obj, page, page_idx, raw_char_blocks_min, table_figure_captions):
    if (page.images == None) or (type(page.images) != list):        ## PJ: Since the Tables are added into this images section, and tables are processed first, here it checks to make sure we don't overwrite any already-extracted tables
        page.images = []
    image_blocks = find_image_blocks(page)

    # A list of all bounding boxes of tables or images found on this page
    allBoundingBoxes = []       ## TODO: Probably has to take page blocks into account, to insert in the right spot?

    allExtractedInfo = []

    print("-"*50)
    print("PAGE:", page_idx)


    # Get only the table/figure captions from this page
    captions_on_page = [c for c in table_figure_captions if c['page_idx'] == page_idx]

    # Table blocks
    table_insert_points = {}
    blocks_to_remove = set()
    pnum = page.pnum

    page_table_boxes = [b for b in page.layout.bboxes if b.label == "Table"]
    page_table_boxes = [rescale_bbox(page.layout.image_bbox, page.bbox, b.bbox) for b in page_table_boxes]
    for table_idx, table_box in enumerate(page_table_boxes):
        for block_idx, block in enumerate(page.blocks):
            intersect_pct = block.intersection_pct(table_box)
            if intersect_pct > settings.BBOX_INTERSECTION_THRESH and block.block_type == "Table":
                if table_idx not in table_insert_points:
                    table_insert_points[table_idx] = block_idx - len(blocks_to_remove) + table_idx # Where to insert the new table
                blocks_to_remove.add(block_idx)

    new_page_blocks = []
    for block_idx, block in enumerate(page.blocks):
        if block_idx in blocks_to_remove:
            continue
        new_page_blocks.append(block)

    for table_idx, table_box in enumerate(page_table_boxes):
        if table_idx not in table_insert_points:
            continue
        # Store the table bounding box in the list of all boxes
        allBoundingBoxes.append(table_box)
        ## TODO: Also store Block?
    print("Added bounding boxes for " + str(len(allBoundingBoxes)) + " tables.")


    # Image blocks
    numImages = 0
    for image_idx, (block_idx, line_idx, bbox) in enumerate(image_blocks):
        if block_idx >= len(page.blocks):
            block_idx = len(page.blocks) - 1
        if block_idx < 0:
            continue

        # Store the table bounding box in the list of all boxes
        allBoundingBoxes.append(bbox)
        ## TODO: Also store Block?  (and any other information?)
        numImages += 1

    print("Added bounding boxes for " + str(numImages) + " images.")
    print("Total number of bounding boxes before merging: " + str(len(allBoundingBoxes)))

    # Merge step: Go through and see if any of the bounding boxes intersect.  If they do, merge them.
    # This is a simple O(n^2) algorithm, but it should be fine for the small number of images/tables we expect.
    print("Merging overlapping table/image bounding boxes...")
    ## TODO: Do we want to do this?
    doneMerge = False
    mergeIterations = 0
    while(not doneMerge):
        numMerged = 0
        for i in range(len(allBoundingBoxes)):
            for j in range(i+1, len(allBoundingBoxes)):
                if (i != j):
                    bbox1 = allBoundingBoxes[i]
                    bbox2 = allBoundingBoxes[j]

                    if (bbox1 is None) or (bbox2 is None):
                        continue

                    # Check if they intersect
                    print("Checking for intersection between " + str(i) + " and " + str(j))
                    print("\t" + str(bbox1))
                    print("\t" + str(bbox2))
                    if bboxIntersects(bbox1, bbox2):
                        # Merge the bounding boxes
                        left = min(bbox1[0], bbox2[0])
                        top = min(bbox1[1], bbox2[1])
                        right = max(bbox1[2], bbox2[2])
                        bottom = max(bbox1[3], bbox2[3])

                        allBoundingBoxes[i] = [left, top, right, bottom]
                        allBoundingBoxes[j] = None
                        print("Merged bounding boxes", bbox1, "and", bbox2, "into", allBoundingBoxes[i])
                        numMerged += 1
                    else:
                        print("No intersection between", i, "and", j)

        # Stop if there (1) were no merges, or (2) we've done too many iterations (signifying the document is complex, or something went wrong)
        if (numMerged == 0):
            doneMerge = True
        mergeIterations += 1
        if (mergeIterations > 20):
            print("WARNING: Merging took too long.  Exiting.")
            doneMerge = True


    # Remove any None values
    allBoundingBoxes = [bbox for bbox in allBoundingBoxes if bbox is not None]
    print("Total number of bounding boxes after merging: " + str(len(allBoundingBoxes)))

    # Match all the bounding boxes to just one caption
    matchedPairs = matchBoundingBoxesToCaptions(allBoundingBoxes, captions_on_page)
    print("Matched pairs:")
    #import json
    #print(json.dumps(matchedPairs, indent=4))
    print(str(matchedPairs))

    # Expand bounding boxes to include captions
    allBoundingBoxesRepacked = []
    for image_idx, bbox1 in enumerate(allBoundingBoxes):
        bbox = bbox1        # Make a copy of the bbox
        print("Examining image " + str(image_idx) + " with bbox " + str(bbox) + " to see if it needs expanding...")

        # First, find it's caption
        #caption, _ = findCaption(bbox, captions_on_page)
        caption = None
        for pair in matchedPairs:
            if pair['bbox_idx'] == image_idx:
                caption = captions_on_page[pair['caption_idx']]
                break

        # PJ: Testing removing the caption merging
        if caption is not None:
            print("Caption found:")
            print(caption)

            print("Expanded bbox to include caption:")
            print("\tImage bbox:", bbox1)
            print("\tCaption bbox:", caption['bbox'])

            # Expand the image bounding box to include the caption
            captionBbox = caption['bbox']
            left = min(bbox[0], captionBbox[0])
            top = min(bbox[1], captionBbox[1])
            right = max(bbox[2], captionBbox[2])
            bottom = max(bbox[3], captionBbox[3])

            bbox = [left, top, right, bottom]
            print("\tExpanded bbox:", bbox)

            # Store the expanded bounding box
            allBoundingBoxes[image_idx] = bbox

        # Re-pack
        packed = {
            'page_idx': page_idx,
            'bbox': bbox,
            'caption': caption
        }
        allBoundingBoxesRepacked.append(packed)

    # Swap, so the list of bounding boxes now has the captions included
    allBoundingBoxes = allBoundingBoxesRepacked


    # Try to re-merge the bounding boxes
    # Merge step: Go through and see if any of the bounding boxes intersect.  If they do, merge them.
    # This is a simple O(n^2) algorithm, but it should be fine for the small number of images/tables we expect.
    print("(2nd merge) Merging overlapping table/image bounding boxes...")
    ## TODO: Do we want to do this?
    doneMerge = False
    mergeIterations = 0
    while(not doneMerge):
        numMerged = 0
        for i in range(len(allBoundingBoxes)):
            for j in range(i+1, len(allBoundingBoxes)):
                if (i != j):
                    bbox1 = allBoundingBoxes[i]['bbox']
                    bbox2 = allBoundingBoxes[j]['bbox']

                    if (bbox1 is None) or (bbox2 is None):
                        continue

                    # Check if they intersect
                    print("Checking for intersection between " + str(i) + " and " + str(j))
                    print("\t" + str(bbox1))
                    print("\t" + str(bbox2))
                    if bboxIntersects(bbox1, bbox2):
                        # Merge the bounding boxes
                        left = min(bbox1[0], bbox2[0])
                        top = min(bbox1[1], bbox2[1])
                        right = max(bbox1[2], bbox2[2])
                        bottom = max(bbox1[3], bbox2[3])

                        allBoundingBoxes[i]['bbox'] = [left, top, right, bottom]
                        allBoundingBoxes[j]['bbox'] = None
                        print("Merged bounding boxes", bbox1, "and", bbox2, "into", allBoundingBoxes[i])
                        numMerged += 1
                    else:
                        print("No intersection between", i, "and", j)

        # Stop if there (1) were no merges, or (2) we've done too many iterations (signifying the document is complex, or something went wrong)
        if (numMerged == 0):
            doneMerge = True
        mergeIterations += 1
        if (mergeIterations > 20):
            print("WARNING: Merging took too long.  Exiting.")
            doneMerge = True

    # Remove any None values
    allBoundingBoxes = [bbox for bbox in allBoundingBoxes if bbox['bbox'] is not None]


    # Now, go through and extract all the images/tables
    #for image_idx, bbox1 in enumerate(allBoundingBoxes):
    # just the range
    for image_idx in range(len(allBoundingBoxes)):
        bbox = allBoundingBoxes[image_idx]['bbox']        # Make a copy of the bbox
        print("Exporting " + str(image_idx) + " with bbox " + str(bbox))

        # First, find it's caption
        caption = allBoundingBoxes[image_idx]['caption']

        ## block = page.blocks[block_idx]
        image = render_bbox_image(page_obj, page, bbox)
        image_filename = get_image_filename(page, image_idx)
        start_marker = "---START_EXTERNAL_IMAGE:ID:" + image_filename + "---"
        end_marker = "---END_EXTERNAL_IMAGE:ID:" + image_filename + "---"
        #image_markdown = f"\n\n![{image_filename}]({image_filename})\n\n"
        image_markdown = f"![{image_filename}]({image_filename})"
        image_markdown = "\n\n" + start_marker + "\n\n" + image_markdown + "\n\n" + end_marker + "\n\n"

        image_span = Span(
            bbox=bbox,
            text=image_markdown,
            font="Image",
            rotation=0,
            font_weight=0,
            font_size=0,
            image=True,
            span_id=f"image_{image_idx}"
        )

        ## PJ (HACKY): Get the last block on this page
        block = page.blocks[-1]     # This might crash if there are no blocks on the page?

        # Sometimes, the block has zero lines
        #if len(block.lines) > line_idx:
        #    block.lines[line_idx].spans.append(image_span)
        #else:
        line = Line(
            bbox=bbox,
            spans=[image_span]
        )
        block.lines.append(line)


        page.images.append(image)

        ## PJ: Try to extract text from the image
        image_text = getTextFromBBox(page_idx, bbox, raw_char_blocks_min)
        if image_text is not None:
            print("IMAGE TEXT:")
            print("PAGE:", page_idx)
            print("BBOX:", bbox)
            # json
            import json
            print(json.dumps(image_text, indent=4))


        # Get the caption text, if easily available
        captionText = ""
        if caption is not None:
            if "text" in caption:
                captionText = caption["text"]

        # Try to infer the extracted type (table or figure) from the caption text
        extractedType = ""
        captionTextNormalized = ""
        if (type(captionText) == str):
            captionTextNormalized = captionText.lower().strip()
        elif (type(captionText) == list):
            captionTextNormalized = " ".join(captionText).lower().strip()

        if (captionTextNormalized.startswith("table")):
            extractedType = "table"
        elif (captionTextNormalized.startswith("figure")):
            extractedType = "figure"

        packed = {
            'page_idx': page_idx,
            'extracted_type': extractedType,        # 'table' or 'figure'
            'bbox': bbox,
            'filename': image_filename,
            'caption': caption,
            'caption_text': captionText,
            'text': image_text,
            'start_marker': start_marker,
            'end_marker': end_marker
        }
        allExtractedInfo.append(packed)

    # Return the extracted info
    return allExtractedInfo


def extract_images(doc, pages, raw_char_blocks_min, table_figure_captions):
    allExtractedInfo = []
    for page_idx, page in enumerate(pages):
        page_obj = doc[page_idx]
        extractedInfo = extract_page_images(page_obj, page, page_idx, raw_char_blocks_min, table_figure_captions)
        allExtractedInfo.extend(extractedInfo)

    return allExtractedInfo




#
## PJ: Helpers
#
def getTextFromBBox(page_idx, bbox, raw_char_blocks_min):
    # Get the page
    if page_idx >= len(raw_char_blocks_min):
        return None
    page = raw_char_blocks_min[page_idx]

    out = []

    for block in page: # page["blocks"]:
        for line in block: # block["lines"]:
            lineText = []

            for span in line: #line["spans"]:
                spanText = span["text"]
                # Make sure it's not an empty string
                if (len(spanText.strip()) == 0):
                    continue

                spanBbox = span["bbox"]
                if bboxIntersects(spanBbox, bbox):
                    lineText.append(spanText)

            if len(lineText) > 0:
                out.append(lineText)

    return out


# Check whether two bounding boxes intersect
def bboxIntersects(bbox1, bbox2):
    # Cases to check: bbox1 is left of bbox2, bbox1 is right of bbox2, bbox1 is above bbox2, bbox1 is below bbox2
    box1Left = bbox1[0]
    box1Right = bbox1[2]
    box1Top = bbox1[1]
    box1Bottom = bbox1[3]

    box2Left = bbox2[0]
    box2Right = bbox2[2]
    box2Top = bbox2[1]
    box2Bottom = bbox2[3]

    # Case 1: bbox1 is left of bbox2
    if box1Right < box2Left:
        return False

    # Case 2: bbox1 is right of bbox2
    if box1Left > box2Right:
        return False

    # Case 3: bbox1 is above bbox2
    if box1Bottom < box2Top:
        return False

    # Case 4: bbox1 is below bbox2
    if box1Top > box2Bottom:
        return False

    # Otherwise, the bounding boxes intersect
    return True


# Rank a set of possible table/figure captions by their likelihood of being a caption for a given table/figure (specified only by it's bbox)
def findCaption(queryBBox, page_captions):
    # queryBBox is a list (with 4 elements) representing the bounding box of the table/figure
    # page_captions is a list of dictionaries, each dictionary representing a caption on the page. it's keys are 'bbox' and 'text'

    if (len(page_captions) == 0):
        return None, None

    captionsScored = []
    # Go through each caption and score it based on it's distance from the queryBBox.
    # If the caption bbox intersects with the queryBBox, it gets a score of 0.
    # Otherwise, the score is the distance between the two bboxes.
    for idx, caption1 in enumerate(page_captions):
        import copy
        caption = copy.deepcopy(caption1)
        captionBBox = caption['bbox']
        caption['idx'] = idx
        if bboxIntersects(queryBBox, captionBBox):
            caption['score'] = 0.0
            captionsScored.append(caption)
        else:
            # Calculate the distance
            dist = bboxDistance(queryBBox, captionBBox)
            caption['score'] = dist
            captionsScored.append(caption)

    # Sort the captions by their score, with the lowest score at index 0
    #captionsScored.sort(key=lambda x: x[1])
    captionsScored.sort(key=lambda x: x['score'])

    # Get the caption with the lowest score
    bestCaption = captionsScored[0]

    return bestCaption, captionsScored


# Match a list of bounding boxes to a list of captions.
# Each caption is matched to a single bounding box with the lowest distance.
def matchBoundingBoxesToCaptions(bboxes, captions):
    if (len(bboxes) == 0) or (len(captions) == 0):
        return []

    # For every caption, score every bounding box
    scores = []
    for cIdx, caption in enumerate(captions):
        captionBBox = caption['bbox']
        scores.append([])
        for bIdx, bbox in enumerate(bboxes):
            dist = bboxDistance(captionBBox, bbox)
            #scores.append((cIdx, bIdx, dist))
            scores[cIdx].append(dist)

    if (len(bboxes) == 1):
        # just find the caption with the lowest score and return it
        minDist = None
        minCaptionIdx = None
        for i in range(len(scores)):
            dist = scores[i][0]
            if minDist is None or dist < minDist:
                minDist = dist
                minCaptionIdx = i
        packed = {
            'caption_idx': minCaptionIdx,
            'bbox_idx': 0,
            'distance': minDist
        }
        return [packed]

    if (len(captions) == 1):
        # just find the bbox with the lowest score and return it
        minDist = None
        minBboxIdx = None
        for i in range(len(scores[0])):
            dist = scores[0][i]
            if minDist is None or dist < minDist:
                minDist = dist
                minBboxIdx = i
        packed = {
            'caption_idx': 0,
            'bbox_idx': minBboxIdx,
            'distance': minDist
        }
        return [packed]



    finalMatches = []
    # Import a constraint solver that minimizes the total distance between the bounding boxes and the captions
    # NOTE: Each caption can only be matched to one bounding box. If there are more bounding boxes than captions, then some bounding boxes will be unmatched.
    #       If there are more captions than bounding boxes, then some captions will be unmatched.
    #       The goal is to minimize the total distance between the matched pairs.
    #       This is a simple assignment problem, and can be solved with the Hungarian algorithm.
    #       The Hungarian algorithm is implemented in the scipy library.
    import numpy as np
    from scipy.optimize import linear_sum_assignment        # pip install scipy
    costMatrix = np.array(scores)
    row_ind, col_ind = linear_sum_assignment(costMatrix)
    # Now, go through and get the matches
    for i in range(len(row_ind)):
        captionIdx = row_ind[i]
        bboxIdx = col_ind[i]
        dist = costMatrix[captionIdx][bboxIdx]
        #finalMatches.append((captionIdx, bboxIdx, dist))
        # Pack
        packed = {
            'caption_idx': captionIdx,
            'bbox_idx': bboxIdx,
            'distance': dist
        }

        finalMatches.append(packed)

    return finalMatches


# Calculate the distance between two bboxes. This does NOT use the centers, but rather the closest points on the two bboxes.
def bboxDistance(bbox1, bbox2):
    # Cases to check: bbox1 is left of bbox2, bbox1 is right of bbox2, bbox1 is above bbox2, bbox1 is below bbox2
    box1Left = bbox1[0]
    box1Right = bbox1[2]
    box1Top = bbox1[1]
    box1Bottom = bbox1[3]

    box2Left = bbox2[0]
    box2Right = bbox2[2]
    box2Top = bbox2[1]
    box2Bottom = bbox2[3]

    # Case 1: bbox1 is left of bbox2
    minDist = None
    if box1Right < box2Left:
        # Distance is between the right of bbox1 and the left of bbox2
        minDist = box2Left - box1Right

    # Case 2: bbox1 is right of bbox2
    if box1Left > box2Right:
        # Distance is between the right of bbox2 and the left of bbox1
        dist = box1Left - box2Right
        if minDist is None or dist < minDist:
            minDist = dist

    # Case 3: bbox1 is above bbox2
    if box1Bottom < box2Top:
        # Distance is between the bottom of bbox1 and the top of bbox2
        dist = box2Top - box1Bottom
        if minDist is None or dist < minDist:
            minDist = dist

    # Case 4: bbox1 is below bbox2
    if box1Top > box2Bottom:
        # Distance is between the bottom of bbox2 and the top of bbox1
        dist = box1Top - box2Bottom
        if minDist is None or dist < minDist:
            minDist = dist

    if (minDist != None):
        return minDist

    # Otherwise, the bounding boxes intersect?
    return 0
