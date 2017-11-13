var configs = {
  domains: ["www.shujuheiban.com"],
  contentUrlRegexes: [/http:\/\/www\.shujuheiban\.com\/.*/], //����ҳurl����
  fields: [
    {
      // ��ȡ��
      name: "code",
      required: true,
      primaryKey : true,
      selector : /.*/,
      selectorType : SelectorType.Regex,
      sourceType : SourceType.UrlContext
    },
    {
      // ��ȡ��
      name: "content",
      selectorType : SelectorType.Regex,
      selector : /.*/
    }
  ]
};
configs.initCrawl = function(site){
  //����ص�����ķ��������첽�ķ�ʽִ��
  //�ⲿ�ı����ͷ�����Ҫ��������ã�����Ҫͨ���ڶ����������ݽ�ȥ
  site.async(function(params){//�����params����async�����ĵڶ�������
    var baseUrl = "http://www.shujuheiban.com/tutorial/lession-1/captcha-generator.php?label=";
    var pickup = "BCEFGHJKMPQRTVWXY2346789";
    var site = params[0];
    var pos = 0;
    var opt = {
      base64 : true
    };
    for(var i =0;i< 100000;i++ ){
      var captcha = "";
      for(var j=0; j<4; j++){
          pos = Math.round(Math.random() * (pickup.length-1));
          captcha += pickup.charAt(pos);
      }
      var url = baseUrl + captcha;
      opt.contextData = captcha; 
      site.addScanUrl(url,opt);
    }
  }, [site]);
};
configs.afterDownloadPage = function (page, site) {
  page.raw = page.raw.replace(/\+/g,"-").replace(/\//g,"_").replace(/\s+/g,"")
  return page;
};
var crawler = new Crawler(configs);
crawler.start();