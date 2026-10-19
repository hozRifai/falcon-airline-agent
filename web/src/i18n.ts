// Tiny bilingual dictionary. The UI language is the person's choice; the assistant's reply language
// follows the language they type in (decided by the runtime), which is a different thing on purpose.
export type Lang = 'en' | 'ar'

const dict = {
  en: {
    appName: 'Falcon Airline', tagline: 'Fictional demo airline',
    signIn: 'Choose who you are', accessCode: 'Access code', enter: 'Enter', wrongCode: 'Wrong access code',
    customer: 'Customer', search: 'Search', trips: 'My trips', assistant: 'Assistant', reason: 'Reason', error: 'Something went wrong',
    from: 'From', to: 'To', dateFrom: 'From date', dateTo: 'To date', passengers: 'Passengers', skip: 'Skip', findFlights: 'Find flights',
    noResults: 'No flights found for that search.', seatsLeft: 'seats left', perPassenger: 'per passenger',
    cheaperFrom: 'Cheaper from', saving: 'saving', drive: 'drive', askAboutFare: 'Ask about this fare',
    special: 'Special', saver: 'Saver', flex: 'Flex', bag: 'Checked bag', changes: 'changes', refund: 'Refund',
    noBag: 'No checked bag', creditOnly: 'credit only', cashRefund: 'credit or card',
    status: { scheduled: 'Scheduled', on_time: 'On time', delayed: 'Delayed', boarding: 'Boarding', departed: 'Departed', landed: 'Landed', cancelled: 'Cancelled', unknown: 'Unknown' } as Record<string, string>,
    bookingStatus: { confirmed: 'Confirmed', cancelled: 'Cancelled', completed: 'Completed' } as Record<string, string>,
    askTrip: 'Ask the assistant about this trip', paid: 'Paid', fare: 'Fare', deadline: 'Cancel/change before',
    typeMessage: 'Type a message in English or Arabic…', send: 'Send', attach: 'Attach a document (PDF or image, up to 2 MB)', attached: 'I have attached a document: {name}', whatIChecked: 'What I checked', thinking: 'Thinking…',
    confirmTitle: 'Please confirm', confirmYes: 'Confirm', confirmNo: 'Not now', declinedNote: 'Nothing was changed.',
    newChat: 'New conversation', signOut: 'Sign out',
    customers: 'Customers', collapse: 'Collapse', expand: 'Expand', focus: 'Focus', focusMode: 'Full-width conversation: hide the side panel', pickCustomer: 'Customer', switching: 'Loading {name}… resetting the demo data and signing in', welcomeTitle: 'Pick a customer on the left to start', welcomeBody: 'Each customer is a situation: a cancelled flight, a missed deadline, a fare that only refunds as credit. Choosing one resets the demo data, signs in as them, and the assistant opens the conversation from what it finds in their bookings. Use Focus for a full-width conversation.', thisCustomer: 'This customer', prefers: 'prefers', sideHint: 'Once a customer is chosen, their trips with live flight status and a flight search appear here: the same data the assistant reads through its tools.',
    verified: 'Checked against', toolResults: 'tool results', amountsChecked: 'amounts verified', reviewSentBack: 'Reviewer sent the draft back', reviewHeld: 'Reply held back by the reviewer', escalated: 'Handed to a human agent after two unresolved replies', reviewing: 'Reviewing the reply…', round: 'round',
    thumbsUp: 'Helpful', thumbsDown: 'Not helpful', thanks: 'Thanks for the feedback', flight: 'Flight', youGetBack: 'You get back', ifApproved: 'If the team approves', noRefund: 'No refund on this fare', fee: 'fee', amountDue: 'Amount due', problems: { unsupported_amounts: 'an amount not from our systems', false_claim: 'claimed an action that did not happen', policy_without_source: 'policy statement without a source', repeated: 'garbled text', announced: 'promised a check without doing it', period_searched_as_one_day: 'searched one day for a period the customer named', pii_in_reply: 'personal data in the reply', prompt_leak: 'repeated its instructions', code_in_reply: 'program code in the reply', wrong_language: 'wrong language', unretrieved_clause: 'cited a clause it had not retrieved', unsupported_quantities: 'a figure the document does not give', reviewer: 'reviewer objection' } as Record<string, string>,
    guardrail: { out_of_scope: 'Outside the assistant\'s scope', prompt_injection: 'Prompt injection blocked', social_engineering: 'Social engineering flagged', abuse: 'Abusive message flagged', redacted: 'Personal data removed before the agent saw it' } as Record<string, string>,
    footer: 'Falcon Airline is fictional and not affiliated with any airline. Policy text: Emirates\' public Conditions of Carriage for Passengers and Baggage, used with credit for demonstration only; fare rules are Falcon\'s own.',
    frozenClock: 'Demo clock', model: 'Model', ms: 'ms', tokens: 'tokens',
    quick: { customer: ['My flight was cancelled, what are my options?', 'Show my bookings', 'I want to cancel my booking', 'Is my flight on time?', 'ما هي سياسة الأمتعة للرضع؟'] } as Record<string, string[]>,
  },
  ar: {
    appName: 'طيران فالكون', tagline: 'شركة طيران تجريبية وهمية',
    signIn: 'اختر هويتك', accessCode: 'رمز الدخول', enter: 'دخول', wrongCode: 'رمز الدخول غير صحيح',
    customer: 'عميل', search: 'بحث', trips: 'رحلاتي', assistant: 'المساعد', reason: 'السبب', error: 'حدث خطأ ما',
    from: 'من', to: 'إلى', dateFrom: 'من تاريخ', dateTo: 'إلى تاريخ', passengers: 'الركاب', skip: 'تجاهل', findFlights: 'ابحث عن رحلات',
    noResults: 'لا توجد رحلات لهذا البحث.', seatsLeft: 'مقاعد متبقية', perPassenger: 'لكل راكب',
    cheaperFrom: 'أرخص من', saving: 'توفير', drive: 'بالسيارة', askAboutFare: 'اسأل عن هذه الدرجة',
    special: 'الخاصة', saver: 'الموفرة', flex: 'المرنة', bag: 'أمتعة مسجلة', changes: 'تغييرات', refund: 'استرداد',
    noBag: 'بدون أمتعة مسجلة', creditOnly: 'رصيد فقط', cashRefund: 'رصيد أو بطاقة',
    status: { scheduled: 'مجدولة', on_time: 'في الموعد', delayed: 'متأخرة', boarding: 'الصعود', departed: 'أقلعت', landed: 'هبطت', cancelled: 'ملغاة', unknown: 'غير معروف' } as Record<string, string>,
    bookingStatus: { confirmed: 'مؤكد', cancelled: 'ملغى', completed: 'مكتمل' } as Record<string, string>,
    askTrip: 'اسأل المساعد عن هذه الرحلة', paid: 'المدفوع', fare: 'الدرجة', deadline: 'الإلغاء/التغيير قبل',
    typeMessage: 'اكتب رسالة بالعربية أو الإنجليزية…', send: 'إرسال', attach: 'إرفاق مستند (PDF أو صورة، حتى 2 م.ب)', attached: 'أرفقت المستند: {name}', whatIChecked: 'ما الذي تحققت منه', thinking: 'جارٍ التفكير…',
    confirmTitle: 'يرجى التأكيد', confirmYes: 'تأكيد', confirmNo: 'ليس الآن', declinedNote: 'لم يتغير شيء.',
    newChat: 'محادثة جديدة', signOut: 'تسجيل الخروج',
    customers: 'العملاء', collapse: 'طيّ', expand: 'توسيع', focus: 'تركيز', focusMode: 'محادثة بكامل العرض: إخفاء اللوحة الجانبية', pickCustomer: 'العميل', switching: 'جارٍ تحميل {name}… إعادة تعيين بيانات العرض وتسجيل الدخول', welcomeTitle: 'اختر عميلاً من القائمة الجانبية للبدء', welcomeBody: 'كل عميل يمثل حالة: رحلة ملغاة، مهلة فائتة، درجة لا تُسترد إلا كرصيد. اختيار عميل يعيد تعيين بيانات العرض ويسجّل الدخول باسمه، ويبدأ المساعد المحادثة مما يجده في حجوزاته. استخدم "تركيز" لمحادثة بكامل العرض.', thisCustomer: 'هذا العميل', prefers: 'يفضّل', sideHint: 'بعد اختيار العميل تظهر هنا رحلاته مع حالة الرحلات المباشرة والبحث عن الرحلات: نفس البيانات التي يقرأها المساعد عبر أدواته.',
    verified: 'تم التحقق مقابل', toolResults: 'نتائج من الأنظمة', amountsChecked: 'مبالغ تم التحقق منها', reviewSentBack: 'أعاد المراجع المسودة', reviewHeld: 'حُجب الرد من قِبل المراجع', escalated: 'أُحيل إلى موظف بعد ردّين لم يُحلّا', reviewing: 'جارٍ مراجعة الرد…', round: 'جولة',
    thumbsUp: 'مفيد', thumbsDown: 'غير مفيد', thanks: 'شكرًا على ملاحظتك', flight: 'الرحلة', youGetBack: 'المبلغ المسترد', ifApproved: 'في حال موافقة الفريق', noRefund: 'لا استرداد لهذه الدرجة', fee: 'الرسوم', amountDue: 'المبلغ المستحق', problems: { unsupported_amounts: 'مبلغ ليس من أنظمتنا', false_claim: 'ادّعاء بإجراء لم يحدث', policy_without_source: 'عبارة من الشروط دون مصدر', repeated: 'نص مشوّش', announced: 'وعد بالتحقق دون تنفيذه', period_searched_as_one_day: 'بحث في يوم واحد عن فترة ذكرها العميل', pii_in_reply: 'بيانات شخصية في الرد', prompt_leak: 'كرّر تعليماته', code_in_reply: 'كود برمجي في الرد', wrong_language: 'لغة خاطئة', unretrieved_clause: 'استشهد ببند لم يسترجعه', unsupported_quantities: 'رقم لا تذكره الوثيقة', reviewer: 'اعتراض المراجع' } as Record<string, string>,
    guardrail: { out_of_scope: 'خارج نطاق المساعد', prompt_injection: 'تم صدّ محاولة حقن تعليمات', social_engineering: 'تم رصد هندسة اجتماعية', abuse: 'تم رصد رسالة مسيئة', redacted: 'أُزيلت بيانات شخصية قبل أن يراها المساعد' } as Record<string, string>,
    footer: 'طيران فالكون شركة وهمية غير مرتبطة بأي شركة طيران. نص الشروط: شروط النقل للمسافرين والأمتعة المنشورة لطيران الإمارات، مستخدمة مع الإشارة للمصدر لأغراض العرض فقط؛ قواعد الأجرة خاصة بفالكون.',
    frozenClock: 'ساعة العرض', model: 'النموذج', ms: 'م.ث', tokens: 'رمز',
    quick: { customer: ['رحلتي ألغيت، ما هي خياراتي؟', 'اعرض حجوزاتي', 'أريد إلغاء حجزي', 'هل رحلتي في موعدها؟', 'ما هي سياسة الأمتعة للرضّع؟'] } as Record<string, string[]>,
  },
}

export type Dict = typeof dict.en
export function dictionary(lang: Lang): Dict { return dict[lang] as Dict }
export const isArabic = (s: string) => /[؀-ۿ]/.test(s) && (s.match(/[؀-ۿ]/g)!.length >= (s.match(/[A-Za-z]/g) || []).length)
