-- Brain diagrams may use the page's full width (Quarto's column-page): wide brains are drawn bigger, and
-- js/life-book.js caps the width of small ones so that their text is never larger than the design size.
function Div(el)
  if el.attributes['life'] == 'brain' or el.attributes['data-life'] == 'brain' then
    el.classes:insert('column-page')
    return el
  end
end
