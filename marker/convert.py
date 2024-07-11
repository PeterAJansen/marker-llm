import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Filter torch pytree user warnings

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS


import pypdfium2 as pdfium # Needs to be at the top to avoid warnings
from PIL import Image

from marker.utils import flush_cuda_memory
from marker.tables.table import format_tables
from marker.debug.data import dump_bbox_debug_data
from marker.layout.layout import surya_layout, annotate_block_types
from marker.layout.order import surya_order, sort_blocks_in_reading_order
from marker.ocr.lang import replace_langs_with_codes, validate_langs
from marker.ocr.detection import surya_detection
from marker.ocr.recognition import run_ocr
from marker.pdf.extract_text import get_text_blocks
from marker.cleaners.headers import filter_header_footer, filter_common_titles
from marker.equations.equations import replace_equations
from marker.pdf.utils import find_filetype
from marker.postprocessors.editor import edit_full_text
from marker.cleaners.code import identify_code_blocks, indent_blocks
from marker.cleaners.bullets import replace_bullets
from marker.cleaners.headings import split_heading_blocks
from marker.cleaners.fontstyle import find_bold_italic
from marker.postprocessors.markdown import merge_spans, merge_lines, get_full_text
from marker.cleaners.text import cleanup_text
from marker.images.extract import extract_images
from marker.images.save import images_to_dict

from typing import List, Dict, Tuple, Optional
from marker.settings import settings


def convert_single_pdf(
        fname: str,
        model_lst: List,
        max_pages: int = None,
        start_page: int = None,
        metadata: Optional[Dict] = None,
        langs: Optional[List[str]] = None,
        batch_multiplier: int = 1,
        subfolder_path: str = ""
) -> Tuple[str, Dict[str, Image.Image], Dict]:
    # Set language needed for OCR
    if langs is None:
        langs = [settings.DEFAULT_LANG]

    if metadata:
        langs = metadata.get("languages", langs)

    langs = replace_langs_with_codes(langs)
    validate_langs(langs)

    # Find the filetype
    filetype = find_filetype(fname)

    # Setup output metadata
    out_meta = {
        "languages": langs,
        "filetype": filetype,
    }

    if filetype == "other": # We can't process this file
        return "", {}, out_meta

    # Get initial text blocks from the pdf
    doc = pdfium.PdfDocument(fname)
    pages, toc, raw_char_blocks = get_text_blocks(      ## PJ: Added RawCharBlocks
        doc,
        fname,
        max_pages=max_pages,
        start_page=start_page
    )
    out_meta.update({
        "toc": toc,
        "pages": len(pages),
    })

    # Identify table/figure captions
    tableFigureCaptions = identify_table_figure_captions(raw_char_blocks)

    # Try to dump the output into a JSON file
    #pathOut = "/home/peter/marker-notes/"
    pathOut = subfolder_path + "/"
    DEBUG_FILENAME = pathOut + "debug.out.json"
    import json
    print("Writing " + DEBUG_FILENAME)
    with open(DEBUG_FILENAME, "w") as f:
        # Pretty print
        #json.dump(pages, f, indent=4)
        json.dump(raw_char_blocks, f, indent=4)

    # Minimize raw_char_blocks
    raw_char_blocks_min, raw_char_blocks_very_min = minimize_raw_char_blocks(raw_char_blocks)
    DEBUG_FILENAME_MIN = pathOut + "debug.min.json"
    print("Writing " + DEBUG_FILENAME_MIN)
    with open(DEBUG_FILENAME_MIN, "w") as f:
        # Pretty print
        #json.dump(pages, f, indent=4)
        json.dump(raw_char_blocks_min, f, indent=4)

    # Output very minimal
    DEBUG_FILENAME_VERY_MIN = pathOut + "debug.min1.json"
    print("Writing " + DEBUG_FILENAME_VERY_MIN)
    with open(DEBUG_FILENAME_VERY_MIN, "w") as f:
        # Pretty print
        #json.dump(pages, f, indent=4)
        json.dump(raw_char_blocks_very_min, f, indent=4)

    # Save just the table/figure captions
    DEBUG_FILENAME_CAPTIONS = pathOut + "debug.captions.json"
    print("Writing " + DEBUG_FILENAME_CAPTIONS)
    with open(DEBUG_FILENAME_CAPTIONS, "w") as f:
        # Pretty print
        #json.dump(pages, f, indent=4)
        json.dump(tableFigureCaptions, f, indent=4)


    # Trim pages from doc to align with start page
    if start_page:
        for page_idx in range(start_page):
            doc.del_page(0)

    # Unpack models from list
    texify_model, layout_model, order_model, edit_model, detection_model, ocr_model = model_lst

    # Identify text lines on pages
    surya_detection(doc, pages, detection_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()

    # OCR pages as needed
    pages, ocr_stats = run_ocr(doc, pages, langs, ocr_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()

    out_meta["ocr_stats"] = ocr_stats
    if len([b for p in pages for b in p.blocks]) == 0:
        print(f"Could not extract any text blocks for {fname}")
        return "", {}, out_meta

    surya_layout(doc, pages, layout_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()

    # Find headers and footers
    bad_span_ids = filter_header_footer(pages)
    out_meta["block_stats"] = {"header_footer": len(bad_span_ids)}

    # Add block types in
    annotate_block_types(pages)

    # Dump debug data if flags are set
    dump_bbox_debug_data(doc, fname, pages)

    # Find reading order for blocks
    # Sort blocks by reading order
    surya_order(doc, pages, order_model, batch_multiplier=batch_multiplier)
    sort_blocks_in_reading_order(pages)
    flush_cuda_memory()

    # Fix code blocks
    code_block_count = identify_code_blocks(pages)
    out_meta["block_stats"]["code"] = code_block_count
    indent_blocks(pages)

    # Fix table blocks
    ## PJ: Commented out this section, and we're now detecting tables as images below.
    table_count = format_tables(doc, pages)
    out_meta["block_stats"]["table"] = table_count
    #out_meta["block_stats"]["table"] = 0

    for page in pages:
        for block in page.blocks:
            block.filter_spans(bad_span_ids)
            block.filter_bad_span_types()

    filtered, eq_stats = replace_equations(
        doc,
        pages,
        texify_model,
        batch_multiplier=batch_multiplier
    )
    flush_cuda_memory()
    out_meta["block_stats"]["equations"] = eq_stats

    # Extract images and figures
    allExtractedTableImageInfo = []
    if settings.EXTRACT_IMAGES:
        allExtractedTableImageInfo = extract_images(doc, pages, raw_char_blocks_min, tableFigureCaptions)

    # Split out headers
    split_heading_blocks(pages)
    find_bold_italic(pages)

    # Copy to avoid changing original data
    merged_lines = merge_spans(filtered)
    text_blocks = merge_lines(merged_lines)
    text_blocks = filter_common_titles(text_blocks)
    full_text = get_full_text(text_blocks)

    # Handle empty blocks being joined
    full_text = cleanup_text(full_text)

    # Replace bullet characters with a -
    full_text = replace_bullets(full_text)

    # Postprocess text with editor model
    full_text, edit_stats = edit_full_text(
        full_text,
        edit_model,
        batch_multiplier=batch_multiplier
    )
    flush_cuda_memory()
    out_meta["postprocess_stats"] = {"edit": edit_stats}
    doc_images = images_to_dict(pages)


    ## PJ: Post processing step 1 -- remove all the baseline extracted tables.  Mark all the table/figure captions with placeholders.
    # Remove baseline exracted tables
    marker_table_start = "----TABLE TEXT START----"
    marker_table_end = "----TABLE TEXT END----"
    done = False
    count = 0
    while (not done):
        startIdx = full_text.find(marker_table_start)
        endIdx = full_text.find(marker_table_end)
        if (startIdx >= 0 and endIdx >= 0):
            # Text before start marker
            beforeText = full_text[:startIdx]
            # Text after end marker
            afterText = full_text[endIdx + len(marker_table_end):]
            # Replace the text between the markers
            full_text = beforeText + afterText
        else:
            done = True

        count += 1
        if (count > 10000):
            print("Error: Infinite loop in table removal. Stopping this process.")
            done = True

    # Add placeholders for table/figure captions
    # Go through the full_text for any paragraphs that start with "Table X:" or "Figure X:", and add a placeholder.
    # The place holder will be "----TABLE X----" or "----FIGURE X----", where X is the number/identifier.
    linesOut = []
    eatActive = False
    insertStr = ""
    for line in full_text.split("\n"):
        if (eatActive):
            # We're eating a multi-line caption
            if (line.strip() == ""):
                eatActive = False
                linesOut.append(insertStr)
                linesOut.append("")
                insertStr = ""
            else:
                lineSanitized = line.strip()
                insertStr += " " + lineSanitized

            continue

        else:
            # Check if the line starts with "Table X:" or "Figure X:"
            import re

            lineSanitized = line.strip().lower()
            lineOut = line
            # Case: Table X: (X is a number)
            match = re.match(r"table [0-9]+:", lineSanitized)
            if match:
                # Extract the number
                tableNumber = match.group(0).split(" ")[1][:-1]
                # Replace the line with a placeholder
                insertStr = "----INSERT TABLE " + tableNumber + " HERE---- " + line
                eatActive = True

            # Case: Figure X: (X is a number)
            match = re.match(r"figure [0-9]+:", lineSanitized)
            if match:
                # Extract the number
                tableNumber = match.group(0).split(" ")[1][:-1]
                # Replace the line with a placeholder
                insertStr = "----INSERT FIGURE " + tableNumber + " HERE---- " + line
                eatActive = True

            if (not eatActive):
                linesOut.append(lineOut)

    full_text = "\n".join(linesOut)



    openaiEnabled = True
    #openaiEnabled = False
    if (openaiEnabled):
        ## PJ: Post-processing step -- try to submit all tables to OpenAI for extraction
        extractedMarkdown = {}
        totalCost = 0
        extractionErrors = 0
        count = 0
        for tableImageInfo in allExtractedTableImageInfo:
            # Dump full text before OpenAI calls
            count += 1
            DEBUG_FILENAME_FULL_TEXT = pathOut + "debug.full_text." + str(count) + ".md"
            print("Writing " + DEBUG_FILENAME_FULL_TEXT)
            with open(DEBUG_FILENAME_FULL_TEXT, "w") as f:
                f.write(full_text)


            # Check the type
            if (tableImageInfo['extracted_type'] == "table"):
                # Get table caption text, if any
                captionText = tableImageInfo['caption_text']
                # Get raw table text, if any
                rawText = tableImageInfo['text']
                # Get the image/filename
                imageFilename = tableImageInfo['filename']
                # Get the image (as PNG?)
                image = None
                if (imageFilename in doc_images.keys()):
                    image = doc_images[imageFilename]

                # Submit for extraction
                if (image is not None):
                    try:
                        extractedTableMarkdown, cost = extract_table_using_LLM(captionText, rawText, image)
                    # Keyboard exception
                    except KeyboardInterrupt:
                        print("Keyboard interrupt.  Stopping.")
                        exit(1)
                    # Other exceptions
                    except Exception as e:
                        import traceback
                        print("Error extracting table using LLM:")
                        print(traceback.format_exc())
                        print(e)

                        extractedTableMarkdown = "** EXTRACTION ERROR **"
                        cost = 0
                        extractionErrors += 1

                    print("Extracted Table Markdown:")
                    print(extractedTableMarkdown)
                    extractedMarkdown[imageFilename] = extractedTableMarkdown
                    totalCost += cost

                    # From the caption text, get the number (e.g. Table X: or Figure X:)
                    captionTextSanitized = captionText
                    if (type(captionText) == list):
                        captionTextSanitized = " ".join(captionText)
                    captionTextSanitized = captionTextSanitized.strip().lower()
                    # Get the X in Table or Figure X:
                    tableNumber = None
                    match = re.match(r"table [0-9]+:", captionTextSanitized)
                    if match:
                        tableNumber = match.group(0).split(" ")[1][:-1]

                    # Check whether the line "INSERT TABLE " + tableNumber + " HERE" exists in the full_text
                    globalMarkerFound = False
                    globalMarker = ""
                    if (tableNumber is not None):
                        globalMarker = "----INSERT TABLE " + str(tableNumber) + " HERE----"
                        for line in full_text.split("\n"):
                            if ("INSERT TABLE " + tableNumber + " HERE" in line):
                                globalMarkerFound = True
                                break

                    # Replace the text in the full_text with the extracted markdown
                    replace_marker_start = tableImageInfo['start_marker']
                    replace_marker_end = tableImageInfo['end_marker']

                    # Replace everything between the start and end marker with the extracted markdown
                    # Find the start and end indices
                    startIdx = full_text.find(replace_marker_start)
                    endIdx = full_text.find(replace_marker_end)
                    extendedMarkdown = ""
                    extendedMarkdown = "\n\n---\n"
                    extendedMarkdown += "**TABLE (AUTOMATICALLY EXTRACTED)**\n\n"
                    extendedMarkdown += extractedTableMarkdown
                    extendedMarkdown += "\n\n"
                    extendedMarkdown += "ERROR: Something happened with automatically detecting start/end markers."
                    extendedMarkdown += "Start marker: `" + replace_marker_start + "`"
                    extendedMarkdown += "End marker: `" + replace_marker_end + "`"
                    extendedMarkdown += "\n\n"
                    extendedMarkdown += "---\n\n"
                    if (startIdx >= 0 and endIdx >= 0):
                        # Text before start marker
                        beforeText = full_text[:startIdx]
                        # Text after end marker
                        afterText = full_text[endIdx + len(replace_marker_end):]

                        # Extract the middle text (should just be an image reference)
                        middleText = full_text[startIdx + len(replace_marker_start):endIdx]

                        # Replace the text between the markers
                        #extractedTableMarkdown = "\n**EXTRACTED USING GPT-4O**\n\n" + extractedTableMarkdown + "\n\n" + middleText + "\n\n"
                        extendedMarkdown = "\n\n---\n"
                        extendedMarkdown += "**TABLE (AUTOMATICALLY EXTRACTED)**\n\n"
                        extendedMarkdown += extractedTableMarkdown
                        extendedMarkdown += "\n\n"
                        extendedMarkdown += middleText
                        extendedMarkdown += "\n\n"
                        extendedMarkdown += "---\n\n"

                        if (not globalMarkerFound):
                            # No global marker found -- just insert it here
                            print("Replacing table in full_text (no global marker found)")
                            full_text = beforeText + extendedMarkdown + afterText
                        else:
                            # Global marker found -- remove this mention
                            full_text = beforeText + "\n" + afterText
                        #full_text = beforeText + extendedMarkdown + afterText

                    if (globalMarkerFound):
                        # Replace the global marker with the table
                        print("Replacing table in full_text (using global marker " + globalMarker + ")")
                        #full_text = full_text.replace(globalMarker, extendedMarkdown)
                        # Search for the index of the global marker

                        # globalMarkerIdx = full_text.find(globalMarker)
                        # if (globalMarkerIdx >= 0):
                        #     # Replace the global marker with the table
                        #     full_text = full_text[:globalMarkerIdx] + extendedMarkdown + full_text[globalMarkerIdx + len(globalMarker):]
                        # else:
                        #     print("Error: Could not find global marker in full_text.")

                        # Look for the line that starts with the globalMarker, and replace that entire line with the table
                        linesOut = []
                        found = False
                        for line in full_text.split("\n"):
                            if (line.strip().startswith(globalMarker)):
                                linesOut.append(extendedMarkdown)
                                found = True
                            else:
                                linesOut.append(line)

                        if (not found):
                            print("Error: Could not find global marker in full_text (`" + str(globalMarker) + "`).")


            elif (tableImageInfo['extracted_type'] == "figure"):
                # Get figure caption text, if any
                captionText = tableImageInfo['caption_text']
                # Get raw figure text, if any
                rawText = tableImageInfo['text']
                # Get the image/filename
                imageFilename = tableImageInfo['filename']
                # Get the image (as PNG?)
                image = None
                if (imageFilename in doc_images.keys()):
                    image = doc_images[imageFilename]

                # Submit for extraction
                if (image is not None):
                    try:
                        extractedFigureMarkdown, cost = extract_figure_using_LLM(captionText, rawText, image)     ### TODO: Change to a figure-specific version.
                    # Keyboard exception
                    except KeyboardInterrupt:
                        print("Keyboard interrupt.  Stopping.")
                        exit(1)
                    # Other exceptions
                    except Exception as e:
                        import traceback
                        print("Error extracting figure using LLM:")
                        print(traceback.format_exc())
                        print(e)

                        extractedFigureMarkdown = "** EXTRACTION ERROR **"
                        cost = 0
                        extractionErrors += 1

                    print("Extracted Figure Markdown:")
                    print(extractedFigureMarkdown)
                    extractedMarkdown[imageFilename] = extractedFigureMarkdown
                    totalCost += cost

                    # From the caption text, get the number (e.g. Table X: or Figure X:)
                    captionTextSanitized = captionText
                    if (type(captionText) == list):
                        captionTextSanitized = " ".join(captionText)
                    captionTextSanitized = captionTextSanitized.strip().lower()
                    # Get the X in Table or Figure X:
                    figureNumber = None
                    match = re.match(r"table [0-9]+:", captionTextSanitized)
                    if match:
                        figureNumber = match.group(0).split(" ")[1][:-1]

                    # Check whether the line "INSERT TABLE " + tableNumber + " HERE" exists in the full_text
                    globalMarkerFound = False
                    globalMarker = ""
                    if (figureNumber is not None):
                        globalMarker = "----INSERT FIGURE " + str(figureNumber) + " HERE----"
                        for line in full_text.split("\n"):
                            if (globalMarker in line):
                                globalMarkerFound = True
                                break

                    # Replace the text in the full_text with the extracted markdown
                    replace_marker_start = tableImageInfo['start_marker']
                    replace_marker_end = tableImageInfo['end_marker']

                    # Replace everything between the start and end marker with the extracted markdown
                    # Find the start and end indices
                    startIdx = full_text.find(replace_marker_start)
                    endIdx = full_text.find(replace_marker_end)
                    extendedMarkdown = ""
                    extendedMarkdown = "\n\n---\n"
                    extendedMarkdown += "**FIGURE (AUTOMATICALLY EXTRACTED)**\n\n"
                    extendedMarkdown += extractedFigureMarkdown
                    extendedMarkdown += "\n\n"
                    extendedMarkdown += "ERROR: Something happened with automatically detecting start/end markers."
                    extendedMarkdown += "Start marker: `" + replace_marker_start + "`"
                    extendedMarkdown += "End marker: `" + replace_marker_end + "`"
                    extendedMarkdown += "\n\n"
                    extendedMarkdown += "---\n\n"
                    if (startIdx >= 0 and endIdx >= 0):
                        # Text before start marker
                        beforeText = full_text[:startIdx]
                        # Text after end marker
                        afterText = full_text[endIdx + len(replace_marker_end):]

                        # Extract the middle text (should just be an image reference)
                        middleText = full_text[startIdx + len(replace_marker_start):endIdx]

                        # Replace the text between the markers
                        #extractedTableMarkdown = "\n**EXTRACTED USING GPT-4O**\n\n" + extractedTableMarkdown + "\n\n" + middleText + "\n\n"
                        extendedMarkdown = "\n\n---\n"
                        extendedMarkdown += "**FIGURE (AUTOMATICALLY EXTRACTED)**\n\n"
                        extendedMarkdown += extractedFigureMarkdown
                        extendedMarkdown += "\n\n"
                        extendedMarkdown += middleText
                        extendedMarkdown += "\n\n"
                        extendedMarkdown += "---\n\n"

                        if (not globalMarkerFound):
                            # No global marker found -- just insert it here
                            print("Replacing figure in full_text (no global marker found)")
                            full_text = beforeText + extendedMarkdown + afterText
                        else:
                            # Global marker found -- remove this mention
                            full_text = beforeText + "\n" + afterText
                        #full_text = beforeText + extendedMarkdown + afterText

                    if (globalMarkerFound):
                        # Replace the global marker with the table
                        print("Replacing figure in full_text (using global marker " + globalMarker + ")")
                        #full_text = full_text.replace(globalMarker, extendedMarkdown)
                        # Search for the index of the global marker
                        globalMarkerIdx = full_text.find(globalMarker)
                        if (globalMarkerIdx >= 0):
                            # Replace the global marker with the table
                            full_text = full_text[:globalMarkerIdx] + extendedMarkdown + full_text[globalMarkerIdx + len(globalMarker):]
                        else:
                            print("Error: Could not find global marker in full_text.")


            else:
                # Do nothing -- unknown type.
                pass

        print("** Total cost of GPT-4 extraction: " + str(totalCost) + " **")

        # Export to JSON
        # DEBUG_FILENAME_EXTRACTED_MARKDOWN = pathOut + "debug.extracted_markdown.json"
        # print("Writing " + DEBUG_FILENAME_EXTRACTED_MARKDOWN)
        # with open(DEBUG_FILENAME_EXTRACTED_MARKDOWN, "w") as f:
        #     # Pretty print
        #     json.dump(extractedMarkdown, f, indent=4)


    return full_text, doc_images, out_meta, allExtractedTableImageInfo



## PJ: Extract table information from images/text
def extract_table_using_LLM(captionText, rawText, image):
    import json
    print("Converting table to markdown using OpenAI...")

    out = {}
    prompt = "I'm working on extracting tables from PDFs, and have (1) an automatically extracted image and (2) automatically extracted text "
    prompt += "(which may have some boundary issues/other small errors).  Can you please convert this table to Markdown, and include the table caption? "
    #prompt += "The only thing you should output is the markdown, as your output is part of an automatic system, and your entire output will be copy/pasted into an existing Markdown document."
    prompt += "You are welcome to think stey-by-step to solve this problem, and provide an explanation of your thought process, but your output should contain only a single set of ``` blocks, where the text inside these will be directly and automatically copy-pasted to the final document, so it *MUST* be correct."
    prompt += "Before you output the extraction in the ``` quotes, please output:\n"
    prompt += "1. How many columns are in this table, as well as the headers for each column.  If there are headers with multi-column spans, break them out -- e.g. if `Model 1` spans two columns (`Precision`, `Recall`), redefine the column header to be (`Precision (Model 1)`, `Recall (Model 1)`).\n"
    prompt += "2. How many rows are in this table?\n"
    prompt += "3. What rows (by number, and description of content) require special formatting, like extra indentation, blank cells, special delimiters, etc., to keep them propery aligned?\n"
    prompt += "\n"

    if (type(captionText) == str):
        prompt += "Automatically Extracted Table Caption: \n```\n" + captionText + "\n```\n"
    elif (type(captionText) == list):
        # JSON pretty print
        captionTextStr = json.dumps(captionText, indent=4)
        prompt += "Automatically Extracted Table Caption: \n```\n" + captionTextStr + "\n```\n"
    else:
        prompt += "Automatically Extracted Table Caption: \n```\n" + str(captionText) + "\n```\n"

    if (type(rawText) == str):
        prompt += "Automatically Extracted Table Text:\n ```\n" + rawText + "\n```\n"
    elif (type(rawText) == list):
        # JSON pretty print
        rawTextStr = json.dumps(rawText, indent=4)
        prompt += "Automatically Extracted Table Text:\n ```\n" + rawTextStr + "\n```\n"
    else:
        prompt += "Automatically Extracted Table Text:\n ```\n" + str(rawText) + "\n```\n"

    prompt += "\n"
    prompt += "Please provide the Markdown representation of the table below (including the table caption, if any, at the bottom of the table)."
    prompt += "Within the ``` quotes, there should be at least one blank line between the table itself and the caption."

    # Submit to OpenAI using litellm
    from litellm import completion

    # Save the image as a PDF
    imageTempFilename = "temp.png"              # NOTE: Obviously not thread safe.
    image.save(imageTempFilename, "PNG")
    # HACKY: Open it and resave it as 2X the size.  Should do this by taking a higher-resolution image of the bbox.
    image = Image.open(imageTempFilename)
    image = image.resize((image.width * 2, image.height * 2))
    image.save(imageTempFilename, "PNG")
    # Reload and encode as base64
    import base64
    base64_image = ""
    with open(imageTempFilename, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    if (len(base64_image) == 0):
        print("Error: Could not encode image as base64.")
        return None


    # Encode image
    #imageEncoded = base64.b64encode(image.read()).decode('utf-8')       # From Hypothesizer
    #imageEncoded = base64.b64encode(image.tobytes()).decode("utf-8")   # Auto-generated

    #messages = [{ "content": "Hello, how are you?","role": "user"}]
    # messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                             {
    #                                 "type": "text",
    #                                 "text": "What’s in this image?"
    #                             },
    #                             {
    #                                 "type": "image_url",
    #                                 "image_url": {
    #                                 "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    #                                 }
    #                             }
    #                         ]
    #         }
    #     ]

    messages=[
        {"role": "user",
         "content": [
            {
                 "type": "text",
                 "text": prompt
            },
            {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",
                    }
            }
         ]
        }
    ]

    response = completion(model="gpt-4o", messages=messages, temperature=0, max_tokens=4000)
    print("\n\n")
    print("-"*50)
    print("\n\n")
    print("Prompt")
    print(prompt)
    print("Response")
    print(response)
    print("\n\n")
    print("-"*50)
    print("\n\n")
    print(response._hidden_params["response_cost"])

    # Get the response text
    responseText = response["choices"][0]["message"]["content"]
    cost = response._hidden_params["response_cost"]
    print("Completed.  Cost: " + str(cost) )

    # Check for leading/trailing ``` lines, and remove them
    responseText = responseText.strip()
    print("RAW RESPONSE TEXT:")
    print("-"*50)
    print(responseText)
    print("-"*50)
    lines = responseText.split("\n")
    # Search for a set of ``` lines
    lineIndices = []
    if (len(lines) >= 2):
        for i in range(len(lines)):
            if (lines[i].strip().startswith("```")):
                lineIndices.append(i)

    if (len(lineIndices) >= 2):
        # take the text between the last two
        secondLastIdx = lineIndices[-2]
        lastIdx = lineIndices[-1]
        responseText = "\n".join(lines[secondLastIdx+1:lastIdx])

    return responseText, cost


## PJ: Extract table information from images/text
def extract_figure_using_LLM(captionText, rawText, image):
    import json
    print("Converting fgure to markdown using OpenAI...")

    out = {}
    prompt = "I'm working on extracting figures from PDFs, and have (1) an automatically extracted image and (2) automatically extracted text "
    prompt += "(which may have some boundary issues/other small errors).  Can you please convert this figure to Markdown, and include the figure caption? "
    prompt += "I'd like you to convert the figure to Markdown in the following way: include two sections, (1) high-level summary, and (2) detailed-summary. "
    prompt += "In the high-level summary, provide a brief overview of the figure, and what the reader should take away from it. "
    prompt += "In the detailed-summary, provide a more detailed explanation of the figure, including any key points, data, or other information that should be taken away. \n"
    #prompt += "The only thing you should output is the markdown, as your output is part of an automatic system, and your entire output will be copy/pasted into an existing Markdown document."
    prompt += "You are welcome to think stey-by-step to solve this problem, and provide an explanation of your thought process, but your output should contain only a single set of ``` blocks, where the text inside these will be directly and automatically copy-pasted to the final document, so it *MUST* be correct."
    prompt += "Before you output the extraction in the ``` quotes, please output a plan in the form of what sections are in the figure, and what you think the user should take away from them."
    prompt += "\n"
    if (type(captionText) == str):
        prompt += "Automatically Extracted Figure Caption: \n```\n" + captionText + "\n```\n"
    elif (type(captionText) == list):
        # JSON pretty print
        captionTextStr = json.dumps(captionText, indent=4)
        prompt += "Automatically Extracted Figure Caption: \n```\n" + captionTextStr + "\n```\n"
    else:
        prompt += "Automatically Extracted Figure Caption: \n```\n" + str(captionText) + "\n```\n"

    if (type(rawText) == str):
        prompt += "Automatically Extracted Figure Text:\n ```\n" + rawText + "\n```\n"
    elif (type(rawText) == list):
        # JSON pretty print
        rawTextStr = json.dumps(rawText, indent=4)
        prompt += "Automatically Extracted Figure Text:\n ```\n" + rawTextStr + "\n```\n"
    else:
        prompt += "Automatically Extracted Figure Text:\n ```\n" + str(rawText) + "\n```\n"

    prompt += "\n"
    prompt += "Please provide the Markdown representation of the figure below (including the figure caption, if any, at the bottom of the figure description)."
    prompt += "Within the ``` quotes, there should be at least one blank line between the figure description itself and the caption."

    # Submit to OpenAI using litellm
    from litellm import completion

    # Save the image as a PDF
    imageTempFilename = "temp.png"              # NOTE: Obviously not thread safe.
    image.save(imageTempFilename, "PNG")
    # HACKY: Open it and resave it as 2X the size.  Should do this by taking a higher-resolution image of the bbox.
    image = Image.open(imageTempFilename)
    image = image.resize((image.width * 2, image.height * 2))
    image.save(imageTempFilename, "PNG")
    # Reload and encode as base64
    import base64
    base64_image = ""
    with open(imageTempFilename, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    if (len(base64_image) == 0):
        print("Error: Could not encode image as base64.")
        return None


    # Encode image
    #imageEncoded = base64.b64encode(image.read()).decode('utf-8')       # From Hypothesizer
    #imageEncoded = base64.b64encode(image.tobytes()).decode("utf-8")   # Auto-generated

    #messages = [{ "content": "Hello, how are you?","role": "user"}]
    # messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                             {
    #                                 "type": "text",
    #                                 "text": "What’s in this image?"
    #                             },
    #                             {
    #                                 "type": "image_url",
    #                                 "image_url": {
    #                                 "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    #                                 }
    #                             }
    #                         ]
    #         }
    #     ]

    messages=[
        {"role": "user",
         "content": [
            {
                 "type": "text",
                 "text": prompt
            },
            {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",
                    }
            }
         ]
        }
    ]

    response = completion(model="gpt-4o", messages=messages, temperature=0, max_tokens=4000)
    print("\n\n")
    print("-"*50)
    print("\n\n")
    print("Prompt")
    print(prompt)
    print("Response")
    print(response)
    print("\n\n")
    print("-"*50)
    print("\n\n")
    print(response._hidden_params["response_cost"])

    # Get the response text
    responseText = response["choices"][0]["message"]["content"]
    cost = response._hidden_params["response_cost"]
    print("Completed.  Cost: " + str(cost) )

    # Check for leading/trailing ``` lines, and remove them
    responseText = responseText.strip()
    print("RAW RESPONSE TEXT:")
    print("-"*50)
    print(responseText)
    print("-"*50)
    lines = responseText.split("\n")
    # Search for a set of ``` lines
    lineIndices = []
    if (len(lines) >= 2):
        for i in range(len(lines)):
            if (lines[i].strip().startswith("```")):
                lineIndices.append(i)

    if (len(lineIndices) >= 2):
        # take the text between the last two
        secondLastIdx = lineIndices[-2]
        lastIdx = lineIndices[-1]
        responseText = "\n".join(lines[secondLastIdx+1:lastIdx])

    return responseText, cost









## PJ: Added helpers
def minimize_raw_char_blocks(raw_char_blocks_in):
    out = []

    outVeryMinimal = []

    # Page
    for page in raw_char_blocks_in:
        out_page = []
        outVeryMinimal_page = []
        # Block
        for block in page["blocks"]:
            out_block = []
            # Line
            for line in block["lines"]:
                out_line = []
                outVeryMinimal_line = []


                # Span
                #out_span = []
                for span in line["spans"]:

                    # Font
                    fontPacked = {}
                    font = span["font"]
                    if "size" in font:
                        fontPacked["size"] = font["size"]
                    if "weight" in font:
                        fontPacked["weight"] = font["weight"]
                    if "name" in font:
                        fontPacked["name"] = font["name"]

                    # Bounding box
                    bbox = {}
                    # convert to left/right/top/bottom
                    bbox["left"] = span["bbox"][0]
                    bbox["right"] = span["bbox"][2]
                    bbox["top"] = span["bbox"][1]
                    bbox["bottom"] = span["bbox"][3]


                    packed = {
                        "text": span["text"],
                        "bbox": span["bbox"],
                        "bbox_named": bbox,
                        "rotation": span["rotation"],
                        "font": fontPacked
                    }

                    # Ensure that the text is not empty
                    if packed["text"].strip() == "":
                        continue

                    #out_line.append(span["text"])
                    out_line.append(packed)
                    justText = span["text"].strip()
                    outVeryMinimal_line.append(justText)

                out_block.append(out_line)
                outVeryMinimal_page.append(outVeryMinimal_line)
            out_page.append(out_block)
        out.append(out_page)
        outVeryMinimal.append(outVeryMinimal_page)

    return out, outVeryMinimal

## PJ: Helper to identify table/figure captions.
def identify_table_figure_captions(raw_char_blocks_in):
    #out = []
    #outVeryMinimal = []
    outCaptions = []

    # Page
    #for page in raw_char_blocks_in:
    for page_idx, page in enumerate(raw_char_blocks_in):
        #out_page = []
        #outVeryMinimal_page = []
        # Block
        for block in page["blocks"]:
            #out_block = []

            isCaptionBlock = False
            captionText = []
            captionBBox = []

            # Line
            for line in block["lines"]:
                #out_line = []
                #outVeryMinimal_line = []

                lineText = ""

                # Span
                #out_span = []
                # Keep track of the min/max extends of the bounding box on this line
                spanBbox = []

                for span in line["spans"]:

                    # Font
                    fontPacked = {}
                    font = span["font"]
                    if "size" in font:
                        fontPacked["size"] = font["size"]
                    if "weight" in font:
                        fontPacked["weight"] = font["weight"]
                    if "name" in font:
                        fontPacked["name"] = font["name"]

                    # Bounding box
                    bbox = {}
                    # convert to left/right/top/bottom
                    bbox["left"] = span["bbox"][0]
                    bbox["right"] = span["bbox"][2]
                    bbox["top"] = span["bbox"][1]
                    bbox["bottom"] = span["bbox"][3]


                    packed = {
                        "text": span["text"],
                        "bbox": span["bbox"],
                        "bbox_named": bbox,
                        "rotation": span["rotation"],
                        "font": fontPacked
                    }

                    # Span bbox
                    spanBbox.append(span["bbox"])

                    # Ensure that the text is not empty
                    lineText += span["text"]
                    if packed["text"].strip() == "":
                        continue

                    #out_line.append(span["text"])
                    #out_line.append(packed)
                    #justText = span["text"].strip()
                    #outVeryMinimal_line.append(justText)

                #out_block.append(out_line)
                #outVeryMinimal_page.append(outVeryMinimal_line)

                # Check if the line text starts with "Table X:" or "Figure X:", where X must be a number (e.g., 1, 2, 3, ..., 10, 11, ...), or a roman numeral (e.g., I, II, III, IV, V, ...), or a single capitol letter (e.g. A, B, C).
                # If so, then we have a table or figure caption.
                import re
                captionStart = False
                # Case: Table X: (X is a number)
                match = re.match(r"Table [0-9]+:", lineText)
                if match:
                    captionStart = True

                # Case: Figure X: (X is a number)
                match = re.match(r"Figure [0-9]+:", lineText)
                if match:
                    captionStart = True

                # Case: Table X: (X is a roman numeral)
                match = re.match(r"Table [IVXLCDM]+:", lineText)
                if match:
                    captionStart = True

                # Case: Figure X: (X is a roman numeral)
                match = re.match(r"Figure [IVXLCDM]+:", lineText)
                if match:
                    captionStart = True

                # Case: Table X: (X is a single capitol letter)
                match = re.match(r"Table [A-Z]:", lineText)
                if match:
                    captionStart = True

                # Case: Figure X: (X is a single capitol letter)
                match = re.match(r"Figure [A-Z]:", lineText)
                if match:
                    captionStart = True

                if captionStart:
                    isCaptionBlock = True

                # If we have a caption block, then we need to keep track of the bounding box of the caption.
                if isCaptionBlock:
                    # Append the caption text of this line to the total caption text
                    captionText.append(lineText)

                    # Find the min/max extents of all spans in the bounding box
                    minLeft = min([bbox[0] for bbox in spanBbox])
                    maxRight = max([bbox[2] for bbox in spanBbox])
                    minTop = min([bbox[1] for bbox in spanBbox])
                    maxBottom = max([bbox[3] for bbox in spanBbox])

                    # Check if the caption bbox is populated.  If not, then populate it.
                    if (len(captionBBox) == 0):
                        captionBBox = [minLeft, minTop, maxRight, maxBottom]
                    else:
                        # Update the caption bbox with new mins/maxes
                        captionBBox[0] = min(captionBBox[0], minLeft)
                        captionBBox[1] = min(captionBBox[1], minTop)
                        captionBBox[2] = max(captionBBox[2], maxRight)
                        captionBBox[3] = max(captionBBox[3], maxBottom)


            # If we reach here and we have a caption block, then we need to add it to the list of captions.
            if isCaptionBlock:
                outCaptions.append({
                    "page_idx": page_idx,
                    "text": captionText,
                    "bbox": captionBBox
                })

#            out_page.append(out_block)
#        out.append(out_page)
#        outVeryMinimal.append(outVeryMinimal_page)
#   return out, outVeryMinimal
    return outCaptions
